from binascii import a2b_base64
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
import matplotlib.pyplot as plt
import camb
import warnings

def scale2(l):
    return l*(l+1)/(2*np.pi)

def scale2deg(l):
    return l*(l+1)/(2*np.pi)*(180/np.pi)**2

def scale4(l):
    return l**4/(2*np.pi)

def getCambSpectra(lmax=7000):
    '''
    Generate and save CMB power spectra using CAMB

    lmax : int
        Maximum multipole (becomes unphysical beyond 7000)
    '''
    pars = camb.CAMBparams()
    # pars.set_accuracy(AccuracyBoost=2)
    pars.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.12, mnu=0.06, omk=0, tau=0.054)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=2) # lens_potential_accuracy=1 is Planck accuracy. See doc
    results = camb.get_results(pars)
    # get dictionary of CAMB power spectra, 'unlensed_scalar' is ordered TT, EE, BB, TE, with BB = 0
    # 'lens_potential' is orderd phi_phi, T_phi, E_phi
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True) # raw_cl so that no factors of ell are multiplied to C_ell
    return powers

def getCaa():
    '''
    For Model II (logarithmically distributed string radii),
    return the theoretical alpha-alpha spectrum
    '''
    dat_arr = np.load('dat/C_aa_model2.npy')
    ls_arr = dat_arr[:,0]
    C_aa_term1_arr = dat_arr[:,1]
    C_aa_term2_arr = dat_arr[:,2]
    C_aa_term1 = interp1d(ls_arr, C_aa_term1_arr, kind='cubic', bounds_error=False, fill_value=0)
    C_aa_term2 = interp1d(ls_arr, C_aa_term2_arr, kind='cubic', bounds_error=False, fill_value=0)
    def C_aa(f_sub):
        def F(l):
            return (1-f_sub)*C_aa_term1(l) + f_sub*C_aa_term2(l)
        return F
    return C_aa

def rotationFromSingleCircularString(N, A, r, shift, parity):
    a_em = 1/137.036 # fine structure constant
    rot_angle = A*a_em
    xs, ys = np.meshgrid(range(N), range(N))
    x0, y0 = shift
    inside = (xs-x0)**2 + (ys-y0)**2 < r**2
    a = rot_angle * inside * parity
    return a

def rotationFromCircularStrings(N, A, r, n):
    shifts = np.random.random((n, 2)) * (N + 2*r) - r # random circle centres
    parities = np.random.randint(0, 2, n) * 2 - 1 # random +-1
    a = np.zeros((N, N))
    for i in range(n):
        a += rotationFromSingleCircularString(N, A, r, shifts[i], parities[i])
    return a

def get_ls(d, N, complex=False):
    '''
    Return real FFT lx, ly coords.
    Convention: first index is y (vertical), second index is x (horizontal)

    d : float
        Pixel width
    N : int
        Number of pixels per side
    '''
    y_freqs = np.fft.fftfreq(N, d) * 2*np.pi
    x_freqs = y_freqs if complex else np.fft.rfftfreq(N, d) * 2*np.pi
    return np.meshgrid(x_freqs, y_freqs)

def modl(lx, ly):
    '''
    lx, ly
        Output of get_ls
    '''
    return np.sqrt(lx**2+ly**2)

def sampleCov(chol):
    '''
    Generate samples of complex Gaussians based on a covariant matrix.
    Return (n, m) complex samples.

    chol
        (n, m, m) array consisting of n m-by-m lower triangular matrix (Cholesky decomposition of the covariant matrix)
        or (n,) array for 1d Gaussian sampling
    '''
    if chol.ndim == 1:
        chol = chol.reshape(chol.size, 1)
    shape = chol.shape[:2] # l-modes, TEBp, real/imag
    normals = (np.random.standard_normal(shape) + 1j*np.random.standard_normal(shape))/np.sqrt(2)
    ret = np.array([chol[i]@normals[i] for i in range(chol.shape[0])])
    return ret

def sampleSpec(d, N, spec, ml=None):
    '''
    Fourier space sampling for a single spectrum, FFT normalisation included

    spec : Spectrum
    '''
    if ml is None:
        ml = modl(*get_ls(d, N))
    ml_flat = ml.reshape(-1)
    chol = np.sqrt(spec(ml_flat))
    ret_flat = sampleCov(chol) * N/d
    return ret_flat.reshape(ml.shape)

def binnedCorrelation(X, Y, delta=20, lmin=None, lmax=None):
    '''
    X, Y : CMBMap
    '''
    prod = (X.f * Y.f.conj()).real * (X.d/X.N)**2
    return Averager(X.d, prod, delta, lmin, lmax)

def plot2Maps(real_map1, real_map2, title1='', title2=''):
    fig, ax = plt.subplots(1, 2, figsize=(16,8))
    ax[0].imshow(real_map1, origin='lower')
    ax[0].set_title(title1)
    ax[1].imshow(real_map2, origin='lower')
    ax[1].set_title(title2)
    plt.show()

class Spectrum:
    '''
    Class containing spectrum function for plotting and generating samples
    '''
    def __init__(self, spec):
        '''
        spec : func
            Power spectrum, function of l, assume already vectorized
        '''
        self.spec = spec

    def __call__(self, l):
        '''
        Spectrum objects can be called directly for evaluation at ls
        '''
        return self.spec(l)

    def __add__(self, other):
        def spec(l):
            return self.spec(l) + other.spec(l)
        return Spectrum(spec)

    def sample(self, l):
        '''
        Generate samples of random complex Gaussians based on spec(ls).
        Variance is spec(l) (factor N**4/W**2 for FFT normalisation is not included)
        Only works if spec(l) > 0.

        ls : 1d array
        '''
        Cls = self.spec(l)
        samples = np.random.standard_normal((l.size, 2))
        ret = (samples[:,0] + 1j*samples[:,1]) * np.sqrt(Cls/2) # /2 for real and imag parts
        return ret

    def generateMap(self, d, N):
        '''
        Return a CMBMap for lensing potential uncorrelated with the unlensed TEB
        '''
        lx, ly = get_ls(d, N)
        ml = modl(lx, ly)    
        ml_flat = ml.reshape(-1)
        p_f = sampleSpec(d, N, self, ml_flat).reshape(ml.shape)
        p_f[0,0] = 0
        return CMBMap(d, N, fourier=p_f)

    def plot(self, ls, scale=lambda l: 1, logaxis=None, **kwargs):
        Cls = self.spec(ls)
        scale_arr = np.vectorize(scale)(ls)
        fmt = kwargs.pop('fmt') if 'fmt' in kwargs else ''
        if logaxis is None:
            plt.plot(ls, Cls * scale_arr, fmt, **kwargs)
        elif logaxis == 'x':
            plt.semilogx(ls, Cls * scale_arr, fmt, **kwargs)
        elif logaxis == 'y':
            plt.semilogy(ls, Cls * scale_arr, fmt, **kwargs)
        elif logaxis == 'both':
            plt.loglog(ls, Cls * scale_arr, fmt, **kwargs)

class InterpSpectrum(Spectrum):
    '''
    Same as Spectrum but initialised using an array
    '''
    def __init__(self, power):
        '''
        power : array_like
            1d array of power spectrum from CAMB
        '''
        self.power = power
        ls = np.arange(power.shape[0])
        self.spec = interp1d(ls, power, bounds_error=False, fill_value=0) # use bounds_error=False, fill_value=0 or not?

class CMBMap:
    '''
    Class containing real space map and real FFT
    '''
    def __init__(self, d, N, fourier=None, real=None):
        self.d = d
        self.N = N
        if fourier is not None:
            self.setFourier(fourier)
        elif real is not None:
            self.setReal(real)

    def copy(self):
        return CMBMap(self.d, self.N, real=self.r.copy())

    def setFourier(self, fourier):
        self.f = fourier # should copy() ?
        self.r = np.fft.irfft2(fourier, s=(self.N, self.N))

    def setReal(self, real):
        self.r = real
        self.f = np.fft.rfft2(real)

    def masked(self, lmin, lmax):
        ml = modl(*self.get_ls())
        lmask = (ml >= lmin) * (ml <= lmax)
        return CMBMap(self.d, self.N, fourier=self.f*lmask)

    def get_ls(self):
        return get_ls(self.d, self.N)

    def binSpectrum(self, delta=20, lmin=None, lmax=None):
        return binnedCorrelation(self, self, delta, lmin, lmax)

    def plot(self, **kwargs):
        plt.imshow(self.r, origin='lower', **kwargs)

class TEB:
    def __init__(self, T, E, B):
        '''
        T, E, B : CMBMap
        '''
        self.d = T.d
        self.N = T.N
        self.T = T
        self.E = E
        self.B = B
        self.lx, self.ly = self.get_ls()
        self.ml = modl(self.lx, self.ly)

    def get_ls(self):
        return get_ls(self.d, self.N)

    def getTQU(self):
        E_f = self.E.f
        B_f = self.B.f
        phi_l = np.arctan2(self.ly, self.lx)
        sin2phi = np.sin(2*phi_l)
        cos2phi = np.cos(2*phi_l)
        Q_f = E_f * cos2phi - B_f * sin2phi
        U_f = E_f * sin2phi + B_f * cos2phi
        Q = CMBMap(self.d, self.N, fourier=Q_f)
        U = CMBMap(self.d, self.N, fourier=U_f)
        return TQU(self.T.copy(), Q, U)

class TQU:
    def __init__(self, T, Q, U):
        '''
        T, Q, U : CMBMap
        '''
        self.d = T.d
        self.N = T.N
        self.T = T
        self.Q = Q
        self.U = U
        self.lx, self.ly = self.get_ls()
        self.ml = modl(self.lx, self.ly)

    def get_ls(self):
        return get_ls(self.d, self.N)

    def getTEB(self):
        Q_f = self.Q.f
        U_f = self.U.f
        phi_l = np.arctan2(self.ly, self.lx)
        sin2phi = np.sin(2*phi_l)
        cos2phi = np.cos(2*phi_l)
        E_f = Q_f * cos2phi + U_f * sin2phi
        B_f = - Q_f * sin2phi + U_f * cos2phi
        E = CMBMap(self.d, self.N, fourier=E_f)
        B = CMBMap(self.d, self.N, fourier=B_f)
        return TEB(self.T.copy(), E, B)

class Averager:
    def __init__(self, d, arr, delta, lmin=None, lmax=None):
        '''
        arr : (N, N//2+1) real array
        d : float
            Pixel width
        delta : float
            Bin width
        '''
        self.d = d
        self.arr = arr
        self.N = arr.shape[0]
        self.delta = delta
        self.ls, self.means, self.stds, self.counts = self.bin(lmin=lmin, lmax=lmax)

    def bin(self, lmin=None, lmax=None):
        lx, ly = get_ls(self.d, self.N)
        ml = modl(lx, ly)
        lmin = 0 if lmin is None else lmin
        lmax = np.max(ml) if lmax is None else lmax

        bounds = np.arange(lmin, lmax, self.delta)
        binned = [self.arr[(ml>=b) * (ml<b+self.delta)] for b in bounds]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            means = np.array([np.mean(b) for b in binned])
            stds = np.array([np.std(b) for b in binned])
        counts = np.array([b.size for b in binned])
        sub = (counts > 0) # remove empty bins

        ls = bounds[sub] + self.delta/2 # bin centres
        means = means[sub]
        stds = stds[sub]
        counts = counts[sub]
        return ls, means, stds, counts

    def plot(self, scale=lambda l: 1, errorbars=True, logaxis=None, **kwargs):
        scales = scale(self.ls)
        if errorbars:
            plt.errorbar(self.ls, self.means * scales, self.stds * scales / np.sqrt(self.counts), **kwargs)
        else:
            if 'fmt' in kwargs.keys():
                fmt = kwargs.pop('fmt')
            plt.plot(self.ls, self.means * scales, fmt, **kwargs)
        if logaxis == 'x' or logaxis == 'both':
            plt.xscale('log')
        if logaxis == 'y' or logaxis == 'both':
            plt.yscale('log')

class CMBSpectra:
    '''
    Class containing spectrum functions and methods for generating samples
    '''
    def __init__(self, powers):
        '''
        powers
            Dict containing power spectra from CAMB
        '''
        self.TT = InterpSpectrum(powers['unlensed_scalar'][:,0])
        self.EE = InterpSpectrum(powers['unlensed_scalar'][:,1])
        self.BB = Spectrum(lambda l: np.zeros(l.shape))
        self.TE = InterpSpectrum(powers['unlensed_scalar'][:,3])
        self.pp = InterpSpectrum(powers['lens_potential'][:,0])
        self.Tp = InterpSpectrum(powers['lens_potential'][:,1])
        self.Ep = InterpSpectrum(powers['lens_potential'][:,2])
        self.TT_len = InterpSpectrum(powers['lensed_scalar'][:,0])
        self.EE_len = InterpSpectrum(powers['lensed_scalar'][:,1])
        self.BB_len = InterpSpectrum(powers['lensed_scalar'][:,2])
        self.TE_len = InterpSpectrum(powers['lensed_scalar'][:,3])

    def choleskyUnlensed(self, ls):
        '''
        Compute the Cholesky decomposition (lower-diagonal square root) of covariance matrix for TE
        '''
        covs = np.zeros((ls.size, 2, 2))
        covs[:,0,0] = self.TT(ls)
        covs[:,1,0] = self.TE(ls)
        covs[:,1,1] = self.EE(ls) # only lower triangular elements used
        chol = np.array([np.linalg.cholesky(c) if np.max(np.abs(c)) > 0 else np.zeros((2, 2)) for c in covs])
        return chol

    def choleskyLensed(self, ls):
        '''
        Compute the Cholesky decomposition (lower-diagonal square root) of covariance matrix for TE lensed
        '''
        covs = np.zeros((ls.size, 2, 2))
        covs[:,0,0] = self.TT_len(ls)
        covs[:,1,0] = self.TE_len(ls)
        covs[:,1,1] = self.EE_len(ls)
        chol = np.array([np.linalg.cholesky(c) if np.max(np.abs(c)) > 0 else np.zeros((2, 2)) for c in covs])
        return chol

    def generatePrimordialMaps(self, d, N):
        '''
        Return a TEB whose FFT modes are sampled from the unlensed spectra stored here

        d : float
            Pixel width
        N : int
            Number of pixels per side
        '''
        lx, ly = get_ls(d, N)
        ml = modl(lx, ly)
        ml_flat = ml.reshape(-1)
        chol = self.choleskyUnlensed(ml_flat)
        samples = sampleCov(chol) * N/d # factor needed for discrete FFT normalisation
        T_f = samples[:,0].reshape(ml.shape)
        T_f[0,0] = 0
        E_f = samples[:,1].reshape(ml.shape)
        E_f[0,0] = 0
        B_f = np.zeros(ml.shape, dtype='complex_')
        T = CMBMap(d, N, fourier=T_f)
        E = CMBMap(d, N, fourier=E_f)
        B = CMBMap(d, N, fourier=B_f)
        return TEB(T, E, B)

    def generateLensingPotential(self, d, N):
        '''
        Return a CMBMap for lensing potential uncorrelated with the unlensed TEB
        '''
        return self.pp.generateMap(d, N)

planck_params = (45, 45*np.sqrt(2), 5)
simons_params = (7, 7*np.sqrt(2), 1.4)
cmb_s4_params = (1, np.sqrt(2), 1.4)

class Detector:
    def __init__(self, D_T, D_P, fwhm):
        '''
        D_T, D_P, fwhm: float
            In arcmin
        '''
        arcmin = 1/60/180*np.pi
        self.D_T = D_T * arcmin
        self.D_P = D_P * arcmin
        self.fwhm = fwhm * arcmin
        self.TTn = Spectrum(self.TTn_fun)
        self.EEn = Spectrum(self.EEn_fun)
        self.BBn = self.EEn

    def TTn_fun(self, l):
        '''
        Beam deconvoluted noise in Fourier space
        '''
        return self.D_T**2 / self.beam(l)**2

    def EEn_fun(self, l):
        return self.D_P**2 / self.beam(l)**2

    def beam(self, l):
        '''
        Beam profile in Fourier space
        '''
        return np.exp(-l**2 * self.fwhm**2 / (16 * np.log(2)))

    def beamConvolve(self, teb):
        beam_f = self.beam(teb.ml)
        T = CMBMap(teb.d, teb.N, fourier=teb.T.f*beam_f)
        E = CMBMap(teb.d, teb.N, fourier=teb.E.f*beam_f)
        B = CMBMap(teb.d, teb.N, fourier=teb.B.f*beam_f)
        return TEB(T, E, B)

    def beamDeconvolve(self, teb):
        beam_f = self.beam(teb.ml)
        T = CMBMap(teb.d, teb.N, fourier=teb.T.f/beam_f)
        E = CMBMap(teb.d, teb.N, fourier=teb.E.f/beam_f)
        B = CMBMap(teb.d, teb.N, fourier=teb.B.f/beam_f)
        return TEB(T, E, B)

    def addNoise(self, teb):
        '''
        Add detector noise to TEB maps. First convolve with beam profile,
        add white noise per real space pixel, then deconvolve with beam profile.
        '''
        tqu = self.beamConvolve(teb).getTQU()
        T_noise_r = np.random.standard_normal((tqu.N, tqu.N)) * self.D_T / tqu.d
        Q_noise_r = np.random.standard_normal((tqu.N, tqu.N)) * self.D_P / tqu.d
        U_noise_r = np.random.standard_normal((tqu.N, tqu.N)) * self.D_P / tqu.d
        T_obs = CMBMap(tqu.d, tqu.N, real=tqu.T.r + T_noise_r)
        Q_obs = CMBMap(tqu.d, tqu.N, real=tqu.Q.r + Q_noise_r)
        U_obs = CMBMap(tqu.d, tqu.N, real=tqu.U.r + U_noise_r)
        tqu_obs = TQU(T_obs, Q_obs, U_obs)
        teb_obs = self.beamDeconvolve(tqu_obs.getTEB())
        return teb_obs

    def addNoiseAlt(self, teb):
        '''
        Directly add noise in Fourier space
        '''
        pass

def grad(p):
    '''
    Calculate gradient field of CMBMap.
    Return two CMBMap's.

    p : CMBMap
        Could be lensing potential
    '''
    lx, ly = p.get_ls()
    delx = CMBMap(p.d, p.N, fourier=p.f * 1j*lx)
    dely = CMBMap(p.d, p.N, fourier=p.f * 1j*ly)
    return delx, dely

def lensInterp(cmbmap, p):
    '''
    Interp
    '''
    d = cmbmap.d
    N = cmbmap.N
    deflx, defly = grad(p) # radians
    inds = np.arange(N)
    indx, indy = np.meshgrid(inds, inds)
    map_interp = RectBivariateSpline(inds, inds, cmbmap.r) # y, x
    new_map = map_interp(indy+defly.r/d, indx+deflx.r/d, grid=False) # pixel units
    return CMBMap(d, N, real=new_map)

def lensTaylor(cmbmap, p):
    '''
    Naive 3rd order Taylor expansion. Bad convergence.
    '''
    d = cmbmap.d
    N = cmbmap.N
    px, py = grad(p)
    mx, my = grad(cmbmap)
    mxx, mxy = grad(mx)
    _, myy = grad(my)
    mxxx, mxxy = grad(mxx)
    _, mxyy = grad(mxy)
    _, myyy = grad(myy)

    order0 = cmbmap.r
    order1 = mx.r*px.r + my.r*py.r
    order2 = 0.5*(mxx.r*px.r*px.r + 2*mxy.r*px.r*py.r + myy.r*py.r*py.r)
    order3 = 1/6*(mxxx.r*px.r**3 + 3*mxxy.r*px.r**2*py.r + 3*mxyy.r*px.r*py.r**2 + myyy.r*py.r**3)

    return CMBMap(d, N, real=order0+order1+order2+order3)

def lensTaylorNearest(cmbmap, p):
    '''
    Return lensed CMBMap given lensing potential

    cmbmap : CMBMap
        Primordial, without noise
    p : CMBMap
        Lensing potential
    '''
    d = cmbmap.d
    N = cmbmap.N
    del1x, del1y = grad(cmbmap)
    del2xx, del2xy = grad(del1x)
    del2yx, del2yy = grad(del1y)

    deflx, defly = grad(p) # deflection field in radians
    inds = np.arange(N)
    indx, indy = np.meshgrid(inds, inds)
    lensedx = indx + deflx.r / d # pixel units
    lensedy = indy + defly.r / d
    nearest_prex = np.rint(lensedx)
    nearest_prey = np.rint(lensedy)
    deltax = (lensedx - nearest_prex) * d # back to radians
    deltay = (lensedy - nearest_prey) * d
    nearestx = (nearest_prex % N).astype(int) # periodic boundary condition
    nearesty = (nearest_prey % N).astype(int)

    lensed_map_real = np.zeros((N, N))
    for i, j in np.ndindex(N, N):
        ind = (nearesty[i,j], nearestx[i,j])
        order0 = cmbmap.r[ind]
        dx = deltax[i,j]
        dy = deltay[i,j]
        order1 = del1x.r[ind]*dx + del1y.r[ind]*dy
        order2 = 0.5 * (del2xx.r[ind]*dx*dx + del2xy.r[ind]*dx*dy + del2yx.r[ind]*dy*dx + del2yy.r[ind]*dy*dy)
        lensed_map_real[i,j] = order0 + order1 + order2
    return CMBMap(d, N, real=lensed_map_real)

def lensTEB(teb, p, fun=lensTaylor):
    '''
    teb : TEB
    p : CMBMap
        Lensing potential
    '''
    tqu = teb.getTQU()
    T_len = fun(tqu.T, p)
    Q_len = fun(tqu.Q, p)
    U_len = fun(tqu.U, p)
    teb_len = TQU(T_len, Q_len, U_len).getTEB()
    return teb_len

def rotateTEB(teb, a):
    tqu = teb.getTQU()
    sin2a = np.sin(2*a.r) # per real space pixel
    cos2a = np.cos(2*a.r)
    Q_r = tqu.Q.r
    U_r = tqu.U.r
    Q_rot_r = Q_r * cos2a - U_r * sin2a
    U_rot_r = Q_r * sin2a + U_r * cos2a
    Q_rot = CMBMap(teb.d, teb.N, real=Q_rot_r)
    U_rot = CMBMap(teb.d, teb.N, real=U_rot_r)
    return TQU(tqu.T, Q_rot, U_rot).getTEB()

def elemTensorProd(A, B):
    '''
    Given A: (m, n, ...) and B: (m, n, ...), multiply A and B in the first two dimensions and preserve all the remaining indices.
    For example, (10, 10, 2, 3) and (10, 10, 5, 6) become (10, 10, 2, 3, 5, 6).
    '''
    inds1 = np.arange(A.ndim)
    inds2 = np.concatenate((np.arange(2), np.arange(A.ndim, A.ndim+B.ndim-2)))
    out = np.concatenate((inds1, inds2[2:]))
    return np.einsum(A, inds1, B, inds2, out)

def expandProd(*factors):
    '''
    Returns a list of 2-tuples that is the foiled-out product of several sums
    Each factor is [(f(l1), g(l2)), ...]
    Each f(l1) or g(l2) is a Fourier map
    '''
    expansion = []
    shape = tuple([len(terms) for terms in factors])
    for inds in np.ndindex(shape):
        new_term1 = factors[0][inds[0]][0].copy()
        new_term2 = factors[0][inds[0]][1].copy()
        for n in range(1, len(inds)):
            new_term1 = elemTensorProd(new_term1, factors[n][inds[n]][0])
            new_term2 = elemTensorProd(new_term2, factors[n][inds[n]][1])
        expansion.append((new_term1, new_term2))
    return expansion

def dotVec(arr, vec):
    '''
    Element-wise dot all remaining dimensions of arr with that of vec

    arr : (m, n, k, ..., k)
    vec : (m, n, k)
    '''
    ret = arr.copy()
    for _ in range(arr.ndim-2):
        ret = np.einsum('ijk...,ijk->ij...', ret, vec)
    return ret

def irfft2(arr):
    '''
    This is for some reason up to 2x faster than np.fft.irfft2(..., axes=(0,1)).
    (np.fft.rfft2 is faster though...)
    '''
    N = arr.shape[0]
    ret_shape = [N, N] + list(arr.shape[2:])
    ret = np.zeros(ret_shape)
    for ind in np.ndindex(arr.shape[2:]):
        slice = tuple([Ellipsis] + list(ind))
        ret[slice] = np.fft.irfft2(arr[slice], s=(N, N))
    return ret

### implement padding here
def convolveTerm(term):
    '''
    term : 2-tuple of (N, N//2+1, ...) arrays
    '''
    fac1_f, fac2_f = term
    fac1_r = np.fft.ifft2(fac1_f, axes=(0, 1))
    fac2_r = np.fft.ifft2(fac2_f, axes=(0, 1))
    prod_r = elemTensorProd(fac1_r, fac2_r)
    prod_f = np.fft.fft2(prod_r, axes=(0, 1))
    return prod_f

def convolveFull(terms, ls):
    '''
    FFT convolve all terms, dotting remaining indices with ls (L)

    terms : list of 2-tuples of (N, N, ...) arrays
    ls : (N, N, 2) array
    '''
    return sum([dotVec(convolveTerm(t), ls) for t in terms])

class Estimator:
    '''
    Some of these methods should be overridden.
    Must implement at least one of f_XY for .qe('XY') or norm('XY') to work.
    '''
    def __init__(self, specs, teb, detector):
        '''
        specs : CMBSpectra
            Power spectra to be used in QEs will be extracted from here
        teb : TEB
        '''
        self.specs = specs
        self.teb = teb
        self.detector = detector
        self.d = teb.d
        self.N = teb.N
        self.lx, self.ly = get_ls(self.d, self.N, complex=True)
        self.ls = np.stack((self.lx, self.ly), -1)
        self.ml = modl(self.lx, self.ly)
        self.ones = np.ones((self.N, self.N))
        self.unnorm_qe = dict() # keys: XY
        self.noise = dict()

    def sin2phi12(self):
        phi = np.arctan2(self.ly, self.lx)
        sin2phi = np.sin(2*phi)
        cos2phi = np.cos(2*phi)
        terms = [
            (sin2phi, cos2phi),
            (-cos2phi, sin2phi)
            ]
        return terms

    def cos2phi12(self):
        phi = np.arctan2(self.ly, self.lx)
        sin2phi = np.sin(2*phi)
        cos2phi = np.cos(2*phi)
        terms = [
            (cos2phi, cos2phi),
            (sin2phi, sin2phi)
            ]
        return terms

    def unnormQETerms(self, XY, lmin, lmax):
        '''
        Quadratic estimator integrand terms, before dotting with L and normalisation.
        Return a list of 2-tuples of complex FFT maps.
        f / (C_TT_tot_l1 * C_TT_tot_l2) / (2 if X==Y else 1), with masking in l space
        '''
        lmask_factor = (self.ml >= lmin) * (self.ml <= lmax)
        lmask = [(lmask_factor, lmask_factor)]
        f_str = 'f_' + XY
        C_XX_t_str = XY[0] + XY[0] + '_t'
        C_YY_t_str = XY[1] + XY[1] + '_t'
        X_r = self.teb.__getattribute__(XY[0]).r
        Y_r = self.teb.__getattribute__(XY[1]).r
        X = np.fft.fft2(X_r) # convert to complex FFT
        Y = np.fft.fft2(Y_r)
        prod = [(X, Y)]
        f = self.__getattribute__(f_str)()
        C_XX_t = self.__getattribute__(C_XX_t_str)
        C_YY_t = self.__getattribute__(C_YY_t_str)
        c = 2 if XY[0] == XY[1] else 1
        denom = [(1/C_XX_t/c, 1/C_YY_t)]
        return expandProd(lmask, prod, denom, f)

    def normTerms(self, XY, lmin, lmax):
        '''
        Normalisation integrand terms, before dotting with L.
        Return a list of 2-tuples of complex FFT maps.
        '''
        lmask_factor = (self.ml >= lmin) * (self.ml <= lmax)
        lmask = [(lmask_factor, lmask_factor)]
        f_str = 'f_' + XY
        C_XX_t_str = XY[0] + XY[0] + '_t'
        C_YY_t_str = XY[1] + XY[1] + '_t'
        f = self.__getattribute__(f_str)()
        C_XX_t = self.__getattribute__(C_XX_t_str)
        C_YY_t = self.__getattribute__(C_YY_t_str)
        c = 2 if XY[0] == XY[1] else 1
        d = self.teb.d
        denom = [(1/C_XX_t/c/d**2, 1/C_YY_t/d**2)] # convert to discrete Fourier before FFT
        return expandProd(lmask, denom, f, f)

    def evaluateQE(self, XY, lmin, lmax):
        '''
        Evaluate the XY lensing estimator as a CMBMap.
        '''
        unnorm_qe_terms = self.unnormQETerms(XY, lmin, lmax)
        norm_terms = self.normTerms(XY, lmin, lmax)
        unnorm_qe = convolveFull(unnorm_qe_terms, self.ls)[:,:self.N//2+1]
        self.unnorm_qe[XY] = unnorm_qe
        norm = convolveFull(norm_terms, self.ls).real[:,:self.N//2+1] * self.d**2 # convert from discrete Fourier to continuous
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            noise = np.nan_to_num(1/norm, posinf=0) # fix /0
            self.noise[XY] = noise
            norm_qe = unnorm_qe * noise
            norm_qe_r = np.fft.irfft2(norm_qe).real
        return CMBMap(self.d, self.N, real=norm_qe_r)

    def plotNoise(self, XY, lmin, lmax, delta=20, **kwargs):
        avg = Averager(self.d, self.noise[XY], delta, lmin, lmax)
        avg.plot(**kwargs)

class LensingEstimator(Estimator):
    '''
    Class that evaluates estimator integrands on the given maps, before plugging them into IndividualEstimator
    '''
    def __init__(self, specs, teb, detector, lensed=True):
        super().__init__(specs, teb, detector)
        # pre-evaluate spectrum at Fourier space points
        if lensed:
            self.TT = specs.TT_len(self.ml)
            self.TE = specs.TE_len(self.ml)
            self.EE = specs.EE_len(self.ml)
            self.BB = specs.BB_len(self.ml)
        else:
            self.TT = specs.TT(self.ml)
            self.TE = specs.TE(self.ml)
            self.EE = specs.EE(self.ml)
            self.BB = specs.BB(self.ml)
        self.TT_t = self.TT + detector.TTn(self.ml)
        self.EE_t = self.EE + detector.EEn(self.ml)
        self.BB_t = self.BB + detector.BBn(self.ml)

    def f_TT(self):
        f = [
            (elemTensorProd(self.TT, self.ls), self.ones),
            (self.ones, elemTensorProd(self.TT, self.ls))
        ]
        return f

    def f_TE(self):
        # approximation of a non-separable integral
        f = expandProd(
            [(elemTensorProd(self.TE, self.ls), self.ones)],
            self.cos2phi12()
        ) + [(self.ones, elemTensorProd(self.TE, self.ls))]
        return f

    def f_TB(self):
        f = expandProd(
            [(elemTensorProd(self.TE, self.ls), self.ones)],
            self.sin2phi12()
        )
        return f

    def f_EE(self):
        f1 = [
            (elemTensorProd(self.EE, self.ls), self.ones),
            (self.ones, elemTensorProd(self.EE, self.ls))
        ]
        f2 = self.cos2phi12()
        return expandProd(f2, f1)

    def f_EB(self):
        f1 = [
            (elemTensorProd(self.EE, self.ls), self.ones),
            (self.ones, -elemTensorProd(self.BB, self.ls))
        ] # remaining index will be dotted with ls (L) after convolution
        f2 = self.sin2phi12()
        return expandProd(f2, f1)

    def f_BB(self):
        f1 = [
            (elemTensorProd(self.BB, self.ls), self.ones),
            (self.ones, elemTensorProd(self.BB, self.ls))
        ]
        f2 = self.cos2phi12()
        return expandProd(f2, f1)

class RotationEstimator(Estimator):
    '''
    Class that evaluates estimator integrands on the given maps, before plugging them into IndividualEstimator
    '''
    def __init__(self, specs, teb, detector, lensed=True):
        super().__init__(specs, teb, detector)
        # pre-evaluate spectrum at Fourier space points
        if lensed:
            self.TT = specs.TT_len(self.ml)
            self.TE = specs.TE_len(self.ml)
            self.EE = specs.EE_len(self.ml)
            self.BB = specs.BB_len(self.ml)
        else:
            self.TT = specs.TT(self.ml)
            self.TE = specs.TE(self.ml)
            self.EE = specs.EE(self.ml)
            self.BB = specs.BB(self.ml)
        self.TT_t = self.TT + detector.TTn(self.ml)
        self.EE_t = self.EE + detector.EEn(self.ml)
        self.BB_t = self.BB + detector.BBn(self.ml)

    def f_TE(self):
        # approximation of a non-separable integral
        f = expandProd(
            [(-2*self.TE, self.ones)],
            self.sin2phi12()
        )
        return f

    def f_TB(self):
        f = expandProd(
            [(2*self.TE, self.ones)],
            self.cos2phi12()
        )
        return f

    def f_EE(self):
        f1 = [
            (-2*self.EE, self.ones),
            (2*self.ones, self.EE)
        ]
        f2 = self.sin2phi12()
        return expandProd(f2, f1)

    def f_EB(self):
        f1 = [
            (2*self.EE, self.ones),
            (-2*self.ones, self.BB)
        ]
        f2 = self.cos2phi12()
        return expandProd(f2, f1)
        
    def f_BB(self):
        f1 = [
            (-2*self.BB, self.ones),
            (2*self.ones, self.BB)
        ]
        f2 = self.sin2phi12()
        return expandProd(f2, f1)