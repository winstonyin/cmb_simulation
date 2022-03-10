import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import camb

def getCambSpectra(lmax=7000):
    '''
    Generate and save CMB power spectra using CAMB

    lmax : int
        Maximum multipole (becomes unphysical beyond 7000)
    '''
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.12, mnu=0.06, omk=0, tau=0.054)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=2) # lens_potential_accuracy=1 is Planck accuracy. See doc
    results = camb.get_results(pars)
    # get dictionary of CAMB power spectra, 'unlensed_scalar' is ordered TT, EE, BB, TE, with BB = 0
    # 'lens_potential' is orderd phi_phi, T_phi, E_phi
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True) # raw_cl so that no factors of ell are multiplied to C_ell
    return powers

def get_ls(d, N):
    '''
    Return real FFT l-coords.
    Convention: first index is y (vertical), second index is x (horizontal)

    d : float
        Pixel width
    N : int
        Number of pixels per side
    '''
    x_freqs = np.fft.rfftfreq(N, d) * 2*np.pi
    y_freqs = np.fft.fftfreq(N, d) * 2*np.pi
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
    '''
    shape = chol.shape[:2] # l-modes, TEBp, real/imag
    normals = (np.random.standard_normal(shape) + 1j*np.random.standard_normal(shape))/np.sqrt(2)
    ret = np.array([chol[i]@normals[i] for i in range(chol.shape[0])])
    return ret

def binnedCorrelation(X, Y, delta=20):
    '''
    X, Y : CMBMap
    '''
    prod = (X.f * Y.f.conj()).real * (X.d/X.N)**2
    return Averager(X.d, prod, delta)

class Spectrum:
    '''
    Class containing spectrum function for plotting and generating samples
    '''
    def __init__(self, spec):
        '''
        cl : func
            Power spectrum, function of l, assume already vectorized
        '''
        self.spec = spec

    def __call__(self, ls):
        '''
        Spectrum objects can be called directly for evaluation at ls
        '''
        return self.spec(ls)

    def sample(self, ls):
        '''
        Generate samples of random complex Gaussians based on spec(ls).
        Variance is spec(l) (factor N**4/W**2 for FFT normalisation is not included)
        Only works if spec(l) > 0.

        ls : 1d array
        '''
        Cls = self.spec(ls)
        samples = np.random.standard_normal((ls.size, 2))
        ret = (samples[:,0] + 1j*samples[:,1]) * np.sqrt(Cls/2) # /2 for real and imag parts
        return ret

    def plot(self, ls, scale=lambda l: 1, **kwargs):
        Cls = self.spec(ls)
        scale_arr = np.vectorize(scale)(ls)
        plt.plot(ls, Cls * scale_arr, **kwargs)

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
        self.spec = interp1d(ls, power) # use bounds_error=False, fill_value=0 or not?

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

    def setFourier(self, fourier):
        self.f = fourier # should copy() ?
        self.r = np.fft.irfft2(fourier)

    def setReal(self, real):
        self.r = real
        self.f = np.fft.rfft2(real)

    def get_ls(self):
        return get_ls(self.d, self.N)

    def binSpectrum(self, delta=20):
        return binnedCorrelation(self, self, delta)

    def plot(self, **kwargs):
        plt.imshow(self.r, **kwargs)

class Averager:
    def __init__(self, d, arr, delta):
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
        self.ls, self.means, self.stds, self.counts = self.bin()

    def bin(self):
        lx, ly = get_ls(self.d, self.arr.shape[0])
        ml = modl(lx, ly)

        bounds = np.arange(0, np.max(ml), self.delta)
        binned = [self.arr[(ml>=b) * (ml<b+self.delta)] for b in bounds]
        means = np.array([np.mean(b) for b in binned])
        stds = np.array([np.std(b) for b in binned])
        counts = np.array([b.size for b in binned])
        sub = (counts > 0) # remove empty bins

        ls = bounds[sub] + self.delta/2 # bin centres
        means = means[sub]
        stds = stds[sub]
        counts = counts[sub]
        return ls, means, stds, counts

    def plot(self, scale=lambda l: 1, errorbars=True, **kwargs):
        scales = scale(self.ls)
        if errorbars:
            plt.errorbar(self.ls, self.means * scales, self.stds * scales / np.sqrt(self.counts), **kwargs)
        else:
            plt.plot(self.ls, self.means * scales, **kwargs)

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
        Compute the Cholesky decomposition (lower-diagonal square root) of covariance matrix for TEp
        '''
        covs = np.zeros((ls.size, 3, 3))
        covs[:,0,0] = self.TT(ls)
        covs[:,1,0] = self.TE(ls)
        covs[:,1,1] = self.EE(ls)
        covs[:,2,0] = self.Tp(ls)
        covs[:,2,1] = self.Ep(ls)
        covs[:,2,2] = self.pp(ls) # only lower triangular elements used
        chol = np.array([np.linalg.cholesky(c) if np.max(np.abs(c)) > 0 else np.zeros((3, 3)) for c in covs])
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
        Returns a TEB and a separate CMBMap for p whose FFT modes are sampled from the unlensed spectra stored here

        d : float
            Pixel width
        N : int
            Number of pixels per side
        '''
        lx, ly = get_ls(d, N)
        ml = modl(lx, ly)
        ml_flat = ml.reshape(-1)
        chol = self.choleskyUnlensed(ml_flat)
        samples = sampleCov(chol) * N/d
        T_f = samples[:,0].reshape(ml.shape)
        E_f = samples[:,1].reshape(ml.shape)
        p_f = samples[:,2].reshape(ml.shape)
        B_f = np.zeros(ml.shape, dtype='complex_')
        T = CMBMap(d, N, fourier=T_f)
        E = CMBMap(d, N, fourier=E_f)
        B = CMBMap(d, N, fourier=B_f)
        p = CMBMap(d, N, fourier=p_f)
        return TEB(T, E, B), p

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

    def get_ls(self):
        return get_ls(self.d, self.N)

    def getTQU(self):
        E_f = self.E.f
        B_f = self.B.f
        lx, ly = self.get_ls()
        phi_l = np.arctan2(ly, lx) # tan(angle) = ly/lx
        sin2phi = np.sin(2*phi_l)
        cos2phi = np.cos(2*phi_l)
        Q_f = E_f * cos2phi - B_f * sin2phi
        U_f = E_f * sin2phi + B_f * cos2phi
        Q = CMBMap(self.d, self.N, fourier=Q_f)
        U = CMBMap(self.d, self.N, fourier=U_f)
        return TQU(self.T, Q, U) # deep copy T?

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

    def get_ls(self):
        return get_ls(self.d, self.N)

    def getTEB(self):
        Q_f = self.Q.f
        U_f = self.U.f
        lx, ly = self.get_ls()
        phi_l = np.arctan2(ly, lx)
        sin2phi = np.sin(2*phi_l)
        cos2phi = np.cos(2*phi_l)
        E_f = Q_f * cos2phi + U_f * sin2phi
        B_f = - Q_f * sin2phi + U_f * cos2phi
        E = CMBMap(self.d, self.N, fourier=E_f)
        B = CMBMap(self.d, self.N, fourier=B_f)
        return TEB(self.T, E, B) # deep copy T?