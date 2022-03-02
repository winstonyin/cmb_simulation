from binascii import a2b_hqx
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import multiprocessing
from multiprocessing import Pool
import camb

multiprocessing.set_start_method('fork') # Necessary for python 3.8 on macOS

planck_params = (45, 45*np.sqrt(2), 5)
simons_params = (7, 7*np.sqrt(2), 1.4)
cmb_s4_params = (1, np.sqrt(2), 1.4)

eps = np.array([[0, -1], [1, 0]]) # -epsilon_ijk where k=z, cross(a,b) = -a@eps@b

def pos(f):
    @np.vectorize
    def g(x):
        return max(0, f(x))
    return g

def get_ls(W, N):
    '''
    Returns N by N//2+1 array of (lx, ly) Fourier space coords
    First side is 0 -- positive modes -- negative modes
    Second side is 0 -- positive modes including Nyquist
    l = 2*np.pi*k/W, where k = 0,...,N-1
    '''
    l0 = np.fft.fftfreq(N, W/N/2/np.pi)
    l1 = np.fft.rfftfreq(N, W/N/2/np.pi)
    return np.array([[(x, y) for y in l1] for x in l0])

def getAngles(ls):
    '''
    Returns N by N//2+1 array of angles of Fourier space vectors returned by get_ls()
    ls: output of get_ls()
    '''
    phi_l = np.zeros(ls.shape[:2])
    for i, j in np.ndindex(phi_l.shape):
        phi_l[i,j] = np.angle(ls[i,j,0]+1j*ls[i,j,1])
    return phi_l

def mirrorFourier(fourier_map):
    '''
    Given a (N, N//2+1) Fourier half-space obtained from rfft2,
    returns the (N, N) Fourier full-space that can be used by ifft2
    '''
    N = fourier_map.shape[0]
    if N%2 == 0:
        zero_mode = np.flip(fourier_map[0,1:-1]).copy()
        to_flip = np.flip(fourier_map[1:,1:-1]).copy()
    else:
        zero_mode = np.flip(fourier_map[0,1:]).copy()
        to_flip = np.flip(fourier_map[1:,1:]).copy()
    second_half = np.concatenate(([zero_mode.conj()], to_flip.conj()), axis=0)
    return np.concatenate((fourier_map, second_half), axis=1)

def mul(a, b):
    '''
    a, b: arrays with the same first 2 dimensions, to be multiplied
    Remaining dimensions of a and b are kept, a first, then b
    '''
    shape = tuple(np.concatenate((a.shape, b.shape[2:])).astype(int))
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        prod = np.zeros(shape, dtype='complex_')
    else:
        prod = np.zeros(shape)
    for ind in np.ndindex(shape[2:]):
        slice0 = tuple([Ellipsis] + list(ind))
        slice1 = tuple([Ellipsis] + list(ind[:a.ndim-2]))
        slice2 = tuple([Ellipsis] + list(ind[a.ndim-2:]))
        prod[slice0] = a[slice1] * b[slice2]
    return prod

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
            new_term1 = mul(new_term1, factors[n][inds[n]][0])
            new_term2 = mul(new_term2, factors[n][inds[n]][1])
        expansion.append((new_term1, new_term2))
    return expansion

def convolveInds(N, L_ind):
    '''
    L_ind: index of fourier mode after fftshift (0 centred)
    Returns pairs (l1_ind, l2_ind) such that their corresponding l1+l2=L
    '''
    if N%2 == 0: # even N
        if L_ind <= N//2-1:
            l1_inds = np.arange(0, L_ind+N//2+1)
        else:
            l1_inds = np.arange(L_ind-N//2+1, N)
    else: # odd N
        if L_ind <= N//2-1:
            l1_inds = np.arange(0, L_ind+N//2+1)
        else:
            l1_inds = np.arange(L_ind-N//2, N)
    l2_inds = np.flip(l1_inds).copy()
    return np.stack((l1_inds, l2_inds), axis=1)

def convolve(fourier_map1, fourier_map2):
    '''
    Given (N, N) fourier maps (already fftshift-ed), returns (N, N) convolution
    '''
    N = fourier_map1.shape[0]
    shape = (N, N)
    if fourier_map1.shape != shape or fourier_map2.shape != shape:
        raise ValueError('Maps have wrong sizes!')
    new_map = np.zeros(shape, dtype='complex_')
    for Lx_ind, Ly_ind in np.ndindex(shape):
        lx_inds = convolveInds(N, Lx_ind)
        ly_inds = convolveInds(N, Ly_ind)
        for l1x_ind, l2x_ind in lx_inds:
            for l1y_ind, l2y_ind in ly_inds:
                new_map[Lx_ind,Ly_ind] += fourier_map1[l1x_ind,l1y_ind]*fourier_map2[l2x_ind,l2y_ind]
    return new_map/N**2

def binSpectrum(cmb_map1, cmb_map2, delta=10):
    '''
    Given two CMBMaps, returns their binned cross power spectrum
    Prefactor of N**4/W**2 due to discretisation is included
    '''
    if cmb_map1.W != cmb_map2.W or cmb_map1.N != cmb_map2.N:
        raise ValueError('The two input maps have different sizes!')
    norm_ls = np.linalg.norm(np.concatenate(get_ls(cmb_map1.W, cmb_map1.N)), axis=1)
    X = np.concatenate(cmb_map1.fourier)
    Y = np.concatenate(cmb_map2.fourier)
    prod = X*np.conj(Y)

    bounds = np.arange(0, np.max(norm_ls), delta)
    binned = [prod[(norm_ls>=b)*(norm_ls<b+delta)] for b in bounds]
    means = np.array([np.mean(b) for b in binned])
    stds = np.array([np.std(b) for b in binned])
    counts = np.array([b.size for b in binned])
    return bounds+delta/2, np.real(means)/cmb_map1.N**4*cmb_map1.W**2, stds/cmb_map1.N**4*cmb_map1.W**2/np.sqrt(counts)

def plot2Maps(real_map1, real_map2, title1='', title2=''):
    fig, ax = plt.subplots(1, 2, figsize=(16,8))
    ax[0].imshow(real_map1)
    ax[0].set_title(title1)
    ax[1].imshow(real_map2)
    ax[1].set_title(title2)
    plt.show()

def lensMapInterp(cmb_map, d):
    '''
    Given a (N, N) real map and a (N, N, 2) deflection field in radians,
    returns the (N, N) lensed real map by interpolation
    d: radians
    '''
    real_map = cmb_map.real
    X = np.arange(real_map.shape[0])
    Y = np.arange(real_map.shape[1])
    f = sp.interpolate.RectBivariateSpline(X, Y, real_map, kx=5, ky=5)
    lensed_coords = np.zeros((cmb_map.N, cmb_map.N, 2))
    for i, j in np.ndindex(real_map.shape):
        lensed_coords[i,j,:] = np.array([i, j]) + d[i,j] *cmb_map.N/cmb_map.W # convert to pixel units
    lensed_map_real = f(lensed_coords[:,:,0], lensed_coords[:,:,1], grid=False)
    return CMBMap(cmb_map.W, cmb_map.N, real_map=lensed_map_real)

def lensMapTaylor(cmb_map, d):
    '''
    Simple 2nd order Taylor expansion X_unlensed(x+d) expanded around d=0 (see https://arxiv.org/pdf/1306.6692.pdf for improvement)
    (Poor convergence at high-L)
    d: radians
    '''
    W = cmb_map.W
    N = cmb_map.N
    ls = get_ls(W, N)
    order1_fourier = tensorProd(1j*cmb_map.fourier, ls)
    order2_fourier = tensorProd(-0.5*cmb_map.fourier, ls, ls)
    order0_real = cmb_map.real
    order1_real = np.fft.irfft2(order1_fourier, s=(N, N), axes=(0, 1))
    order2_real = np.fft.irfft2(order2_fourier, s=(N, N), axes=(0, 1))
    lensed_map_real = order0_real + dotVec(order1_real, d, complex=False) + dotVec(order2_real, d, complex=False)
    return CMBMap(W, N, real_map=lensed_map_real)

def lensMapTaylorNearest(cmb_map, d):
    '''
    Simple 2nd order Taylor expansion X_unlensed(x+d) expanded around d=nearest_pixel
    '''
    W = cmb_map.W
    N = cmb_map.N
    ls = get_ls(W, N)
    order1_fourier = tensorProd(1j*cmb_map.fourier, ls)
    order2_fourier = tensorProd(-0.5*cmb_map.fourier, ls, ls)
    order0_real = cmb_map.real
    order1_real = np.fft.irfft2(order1_fourier, s=(N, N), axes=(0, 1))
    order2_real = np.fft.irfft2(order2_fourier, s=(N, N), axes=(0, 1))
    d_pixel = d*N/W # pixel units
    lensed_map_real = np.zeros((N, N))
    for i, j in np.ndindex(N, N):
        lensed_coords = np.array([i+d_pixel[i,j,0], j+d_pixel[i,j,1]])
        nearest_coords_pre = np.rint(lensed_coords).astype(int)
        delta = (lensed_coords - nearest_coords_pre)*W/N # back to radians
        nearest_coords = tuple(nearest_coords_pre % N) # periodic boundary
        lensed_map_real[i,j] = order0_real[nearest_coords] + order1_real[nearest_coords]@delta + order2_real[nearest_coords]@delta@delta
    return CMBMap(W, N, real_map=lensed_map_real)

def tensorProd(first, *vecs):
    '''
    first: (m, n)
    vecs: each is (m, n, n_i)
    returns: (m, n, n_0, ..., n_imax) by multiplication element-wise in the first two dimensions
    '''
    for v in vecs:
        if v.shape[:2] != first.shape or v.ndim != 3:
            raise ValueError('Incompatible shapes!')
    shape = list(first.shape)
    for v in vecs:
        shape.append(v.shape[2])
    prod = np.zeros(shape, dtype='complex_')
    for ind in np.ndindex(tuple(shape)):
        vals = [first[ind[:2]]]
        for i, v in enumerate(vecs):
            vals.append(v[ind[0],ind[1],ind[2+i]])
        prod[ind] = np.prod(vals)
    return prod

def dotVec(tensor, vec, complex=True):
    '''
    tensor: (N, N//2+1, n, ..., n)
    Returns (N, N//2+1) by dotting all remaining dimensions with vec element-wise
    '''
    if complex:
        result = np.zeros(tensor.shape[:2], dtype='complex_')
    else:
        result = np.zeros(tensor.shape[:2])
    for ind in np.ndindex(result.shape):
        L = vec[ind]
        subresult = tensor[ind].copy()
        for i in range(tensor.ndim-2):
            subresult = subresult@L
        result[ind] = subresult
    return result

class CMBMap:
    def __init__(self, W, N, fourier_map=None, real_map=None):
        self.W = W
        self.N = N
        if fourier_map is not None:
            self.setFourier(fourier_map)
        elif real_map is not None:
            self.setReal(real_map)

    def setFourier(self, fourier_map):
        '''
        fourier_map: (N, N//2+1) for use by irfft2
        '''
        self.fourier = fourier_map
        self.real = np.fft.irfft2(fourier_map, s=(self.N, self.N))

    def setReal(self, real_map):
        self.real = real_map
        self.fourier = np.fft.rfft2(real_map)

    def __add__(self, b):
        return CMBMap(self.W, self.N, fourier_map=self.fourier+b.fourier)

    def __sub__(self, b):
        return CMBMap(self.W, self.N, fourier_map=self.fourier-b.fourier)

    def conj(self):
        return CMBMap(self.W, self.N, fourier_map=np.conj(self.fourier))

class CMBGenerator:
    def __init__(self, powers, W, N, lmax=7000):
        '''
        powers: a dict generated by CAMB
        W: side length of the square in radians
        N: number of pixels on each side
        lmax: modes are assumed to be 0 beyond lmax
        '''
        pw = powers['unlensed_scalar']
        lens_pw = powers['lensed_scalar']
        p_pw = powers['lens_potential']
        ls = np.arange(lmax+1)
        self.C_TT = sp.interpolate.interp1d(ls, pw[:lmax+1,0], bounds_error=False, fill_value=0) # interpolate power spectra from CAMB
        self.C_TE = sp.interpolate.interp1d(ls, pw[:lmax+1,3], bounds_error=False, fill_value=0)
        self.C_EE = sp.interpolate.interp1d(ls, pw[:lmax+1,1], bounds_error=False, fill_value=0)
        self.C_pp = sp.interpolate.interp1d(ls, p_pw[:lmax+1,0], bounds_error=False, fill_value=0)
        self.C_Tp = sp.interpolate.interp1d(ls, p_pw[:lmax+1,1], bounds_error=False, fill_value=0)
        self.C_Ep = sp.interpolate.interp1d(ls, p_pw[:lmax+1,2], bounds_error=False, fill_value=0)
        self.C_TT_lensed = sp.interpolate.interp1d(ls, lens_pw[:lmax+1,0], bounds_error=False, fill_value=0)
        self.C_TE_lensed = sp.interpolate.interp1d(ls, lens_pw[:lmax+1,3], bounds_error=False, fill_value=0)
        self.C_EE_lensed = sp.interpolate.interp1d(ls, lens_pw[:lmax+1,1], bounds_error=False, fill_value=0)
        self.C_BB_lensed = sp.interpolate.interp1d(ls, lens_pw[:lmax+1,2], bounds_error=False, fill_value=0)
        self.W = W
        self.N = N
        self.lmax = lmax
        # if 2*np.pi*(N-1)/W > np.max(ls):
        #     raise ValueError('N/side_length too large for the generated multipoles up to l=' + str(np.max(ls)))
        self.ls_grid = get_ls(W, N)
        self.norm_ls_grid = np.array([[np.linalg.norm(l) for l in row] for row in self.ls_grid])

    def sampleCov(self, norm_l):
        '''
        Returns random complex numbers T(l), E(l), a(l) in discrete Fourier space whose covariance matrix is
        N**4/W**2 * [[C_TT(l), C_TE(l), C_Tp(l)], [C_TE(l), C_EE(l), C_Ep(l)], [C_Tp(l), C_Ep(l), C_pp(l)]]
        norm_l: magnitude of vector l
        '''
        if norm_l > self.lmax:
            return np.zeros(3, dtype='complex_')
        Cmat = np.array([
            [self.C_TT(norm_l), self.C_TE(norm_l), self.C_Tp(norm_l)],
            [self.C_TE(norm_l), self.C_EE(norm_l), self.C_Ep(norm_l)],
            [self.C_Tp(norm_l), self.C_Ep(norm_l), self.C_pp(norm_l)]
            ]) # covariance matrix for T, E, phi
        w, v = np.linalg.eigh(Cmat) # diagonalise Cmat
        stdev = np.sqrt(np.maximum(0, w)/2)*self.N**2/self.W # half variance for real part and half for imag part
        z = np.array([
            np.random.normal(scale=stdev[i]) + 1.j*np.random.normal(scale=stdev[i]) for i in range(3)
        ]) # generate samples in eigenbasis
        return v@z # basis transform from eigenbasis to T, E, phi

    def sample(self, spec, norm_l, cap_lmax=True):
        '''
        spec: interpolated or continuous power spectrum function
        norm_l: magnitude of vector l
        Returns a complex Gaussian sample with variance N**4/W**2* spec(l)
        '''
        if cap_lmax and norm_l > self.lmax:
            return np.zeros(1, dtype='complex_')
        stdev = np.sqrt(spec(norm_l)/2)*self.N**2/self.W
        return np.random.normal(scale=stdev) + 1.j*np.random.normal(scale=stdev)

    def setPrimordialMaps(self):
        '''
        Sets primordial CMB maps T, E, B, phi without any noise
        '''
        samples = np.zeros((self.N, self.N//2+1, 3), dtype='complex_')
        for i, j in np.ndindex(self.N, self.N//2+1):
            if i == 0 or j == 0 or (self.N % 2 == 0 and (i == self.N//2 or j == self.N//2)):
                samples[i,j,:] = (0,0,0) # ignore 0 and Nyquist modes
            else:
                samples[i,j,:] = self.sampleCov(np.linalg.norm(self.ls_grid[i,j]))
        self.T_prim = CMBMap(self.W, self.N, fourier_map=samples[:,:,0])
        self.E_prim = CMBMap(self.W, self.N, fourier_map=samples[:,:,1])
        self.p = CMBMap(self.W, self.N, fourier_map=samples[:,:,2])
        self.B_prim = CMBMap(self.W, self.N, fourier_map=np.zeros((self.N, self.N//2+1), dtype='complex_'))

    def C_TT_n(self, l):
        const = self.fwhm**2/(8*np.log(2))
        return self.D_T**2*np.exp(l**2*const)

    def C_EE_n(self, l):
        const = self.fwhm**2/(8*np.log(2))
        return self.D_P**2*np.exp(l**2*const)

    C_BB_n = C_EE_n

    def setNoiseMaps(self, D_T, D_P, fwhm):
        '''
        D_T, D_P given in muK arcmin
        fwhm given in arcmin
        Planck: 45, 45*np.sqrt(2), 5
        Simons: 7, 7*np.sqrt(2), 1.4
        CMB-S4: 1, np.sqrt(2), 1.4
        '''
        self.D_T = D_T*np.pi/180/60 # unit conversion to radians
        self.D_P = D_P*np.pi/180/60
        self.fwhm = fwhm*np.pi/180/60

        samples = np.zeros((self.N, self.N//2+1, 3), dtype='complex_')
        for i, j in np.ndindex(self.N, self.N//2+1):
            if i == 0 or j == 0 or (self.N % 2 == 0 and (i == self.N//2 or j == self.N//2)):
                samples[i,j,:] = (0,0,0)
            else:
                samples[i,j,0] = self.sample(self.C_TT_n, np.linalg.norm(self.ls_grid[i,j]), cap_lmax=False)
                samples[i,j,1] = self.sample(self.C_EE_n, np.linalg.norm(self.ls_grid[i,j]), cap_lmax=False)
                samples[i,j,2] = self.sample(self.C_BB_n, np.linalg.norm(self.ls_grid[i,j]), cap_lmax=False)
        self.T_noise = CMBMap(self.W, self.N, fourier_map=samples[:,:,0])
        self.E_noise = CMBMap(self.W, self.N, fourier_map=samples[:,:,1])
        self.B_noise = CMBMap(self.W, self.N, fourier_map=samples[:,:,2])

    def EB2QU(self, E_map, B_map):
        '''
        Given CMBMaps E and B, returns CMBMaps Q and U
        '''
        E = E_map.fourier
        B = B_map.fourier
        if E.shape != B.shape:
            raise ValueError('E and B have different shapes!')
        phi = np.zeros(self.ls_grid.shape[:2])
        for i, j in np.ndindex(phi.shape):
            phi[i,j] = np.angle(self.ls_grid[i,j,0]+1j*self.ls_grid[i,j,1])
        cos2phi = np.cos(2*phi)
        sin2phi = np.sin(2*phi)
        Q = E*cos2phi - B*sin2phi
        U = E*sin2phi + B*cos2phi
        Q_map = CMBMap(self.W, self.N, fourier_map=Q)
        U_map = CMBMap(self.W, self.N, fourier_map=U)
        return Q_map, U_map

    def QU2EB(self, Q_map, U_map):
        '''
        Given CMBMaps Q and U, returns CMBMaps E and B
        '''
        Q = Q_map.fourier
        U = U_map.fourier
        if Q.shape != U.shape:
            raise ValueError('Q and U have different shapes!')
        phi = np.zeros(self.ls_grid.shape[:2])
        for i, j in np.ndindex(phi.shape):
            phi[i,j] = np.angle(self.ls_grid[i,j,0]+1j*self.ls_grid[i,j,1])
        cos2phi = np.cos(2*phi)
        sin2phi = np.sin(2*phi)
        E = Q*cos2phi + U*sin2phi
        B = -Q*sin2phi + U*cos2phi
        E_map = CMBMap(self.W, self.N, fourier_map=E)
        B_map = CMBMap(self.W, self.N, fourier_map=B)
        return E_map, B_map

    def deflection(self):
        '''
        Returns the (N, N, 2) deflection vector field in real space
        '''
        d_fourier = tensorProd(self.p.fourier, 1j*self.ls_grid)
        d_real = np.fft.irfft2(d_fourier, s=(self.N, self.N), axes=(0, 1))
        return d_real

    def setLensedMaps(self, mode='nearest'):
        '''
        Saves the lensed maps of primordial modes plus noise
        '''
        d = self.deflection()
        if mode == 'taylor':
            lensMap = lensMapTaylor
        elif mode == 'nearest':
            lensMap = lensMapTaylorNearest
        elif mode == 'interp':
            lensMap = lensMapInterp
        else:
            raise ValueError('Unrecognised mode...')

        self.T_lensed = lensMap(self.T_prim, d)

        Q, U = self.EB2QU(self.E_prim, self.B_prim)
        Q_lensed = lensMap(Q, d)
        U_lensed = lensMap(U, d)
        self.E_lensed, self.B_lensed = self.QU2EB(Q_lensed, U_lensed)

    def setRotationMap(self, C_aa):
        samples = np.zeros((self.N, self.N//2+1), dtype='complex_')
        for i, j in np.ndindex(self.N, self.N//2+1):
            if i == 0 or j == 0 or (self.N % 2 == 0 and (i == self.N//2 or j == self.N//2)):
                samples[i,j] = 0
            else:
                samples[i,j] = self.sample(C_aa, np.linalg.norm(self.ls_grid[i,j]))
        self.a = CMBMap(self.W, self.N, fourier_map=samples)

    def setRotatedMaps(self):
        self.T_rot = self.T_lensed
        Q, U = self.EB2QU(self.E_lensed, self.B_lensed)
        cos2a = np.cos(2*self.a.real)
        sin2a = np.sin(2*self.a.real)
        Q_rot_real = Q.real*cos2a - U.real*sin2a
        U_rot_real = Q.real*sin2a + U.real*cos2a
        Q_rot = CMBMap(self.W, self.N, real_map=Q_rot_real)
        U_rot = CMBMap(self.W, self.N, real_map=U_rot_real)
        self.E_rot, self.B_rot = self.QU2EB(Q_rot, U_rot)

    def convolveIntegralTerm(self, factors):
        '''
        Each factor: 2-tuple of ndarrays (N, N//2+1, 2, ..., 2) in discrete Fourier space
        Returns the discrete Fourier space convolution integral (single term) between them as (N, N//2+1)
        by multiplying their real space maps and then Fourier transformed,
        where remaining dimensions are dotted with L
        '''
        factor1_real = np.fft.irfft2(factors[0], s=(self.N, self.N), axes=(0,1))
        factor2_real = np.fft.irfft2(factors[1], s=(self.N, self.N), axes=(0,1))
        free_dims = factor1_real.ndim - factor2_real.ndim # excess dimensions to be dotted with L, assumed to all be in one factor
        shape = [2] * (abs(free_dims)+2)
        shape[:2] = (self.N, self.N)
        shape = tuple(shape)
        integral_real = np.zeros(shape)
        for ind in np.ndindex(shape):
            if free_dims > 0:
                slice1 = tuple(np.concatenate((ind[:2], [...], ind[-free_dims:])))
                slice2 = tuple(np.concatenate((ind[:2], [...])))
            elif free_dims < 0:
                slice1 = tuple(np.concatenate((ind[:2], [...])))
                slice2 = tuple(np.concatenate((ind[:2], [...], ind[free_dims:])))
            else:
                slice1 = tuple(np.concatenate((ind[:2], [...])))
                slice2 = tuple(np.concatenate((ind[:2], [...])))
            integral_real[ind] = np.sum(factor1_real[slice1]*factor2_real[slice2])
        integral_fourier = np.fft.rfft2(integral_real, axes=(0, 1))
        result = dotVec(integral_fourier, self.ls_grid)
        return result

    def convolveIntegral(self, factors):
        '''
        Returns one full convolution integral (all terms summed)
        factors: list of 2-tuples of individual factors
        '''
        with Pool() as pool:
            terms = pool.map(self.convolveIntegralTerm, factors)
        return sum(terms)

    def lensingEstimator(self, factors_est, factors_norm):
        integral_est_fourier = self.convolveIntegral(factors_est)
        integral_norm_fourier = self.convolveIntegral(factors_norm) * (self.W/self.N)**2 # divide by continuous N(L), not discrete Fourier N_L
        p_est_fourier = np.zeros((self.N, self.N//2+1), dtype='complex_')
        for i, j in np.ndindex(p_est_fourier.shape):
            if i == 0 and j == 0:
                p_est_fourier[i,j] == 0
            else:
                p_est_fourier[i,j] = integral_est_fourier[i,j]/integral_norm_fourier[i,j]
        return CMBMap(self.W, self.N, fourier_map=p_est_fourier)

    rotationEstimator = lensingEstimator

    def lensing_EB(self, null=False):
        '''
        Factors of ls to be dottted with Ls must be towards the end (right)
        null: True if there's no lensing in the input maps, False if there is
        '''
        ls = self.ls_grid.copy()
        norm_ls = self.norm_ls_grid.copy()
        norm_ls[0,0] = 1e-10 # to avoid /0
        C_EE = self.C_EE(norm_ls)
        C_EE_n = self.C_EE_n(norm_ls)
        C_BB_n = self.C_BB_n(norm_ls)
        phi_l = getAngles(ls)
        cos2phi = np.cos(2*phi_l)
        sin2phi = np.sin(2*phi_l)
        if null:
            E = self.E_prim.fourier + self.E_noise.fourier
            B = self.B_prim.fourier + self.B_noise.fourier
        else:
            E = self.E_lensed.fourier + self.E_noise.fourier
            B = self.B_lensed.fourier + self.B_noise.fourier
        ones = np.ones((self.N, self.N//2+1))

        qe = [(E, B)]
        f_factor1 = [(tensorProd(2 * C_EE, ls), ones)] # put this factor towards the end for dotting with L
        f_factor2 = [(sin2phi, cos2phi), (-cos2phi, sin2phi)]
        denom = [(1 / (C_EE + C_EE_n), 1 / C_BB_n)]
        convert = [(ones * (self.N/self.W)**2, ones * (self.N/self.W)**2)] # need to convert to discrete Fourier before FFT convolve

        factors_est = expandProd(qe, denom, f_factor2, f_factor1)
        factors_norm = expandProd(denom, convert, f_factor2, f_factor2, f_factor1, f_factor1)

        return factors_est, factors_norm

    def lensing_EE(self, null=False):
        ls = self.ls_grid.copy()
        norm_ls = self.norm_ls_grid.copy()
        norm_ls[0,0] = 1e-10 # to avoid /0
        C_EE = self.C_EE(norm_ls)
        C_EE_n = self.C_EE_n(norm_ls)
        phi_l = getAngles(ls)
        cos2phi = np.cos(2*phi_l)
        sin2phi = np.sin(2*phi_l)
        if null:
            E = self.E_prim.fourier + self.E_noise.fourier
        else:
            E = self.E_lensed.fourier + self.E_noise.fourier
        ones = np.ones((self.N, self.N//2+1))

        qe = [(E, E)]
        f_factor1 = [(tensorProd(C_EE, ls), ones), (ones, tensorProd(C_EE, ls))]
        f_factor2 = [(cos2phi, cos2phi), (sin2phi, sin2phi)]
        denom = [(1 / (C_EE + C_EE_n), 1 / (C_EE + C_EE_n))]
        convert = [(ones * (self.N/self.W)**2, ones * (self.N/self.W)**2)]

        factors_est = expandProd(qe, denom, f_factor2, f_factor1)
        factors_norm = expandProd(denom, convert, f_factor2, f_factor2, f_factor1, f_factor1)

        return factors_est, factors_norm

    def rotation_EB(self, null=False, lensed=True):
        '''
        null: True if there's no rotation in the input maps, False if there is
        lensed: whether the input maps are lensed
        '''
        ls = self.ls_grid.copy()
        norm_ls = self.norm_ls_grid.copy()
        norm_ls[0,0] = 1e-10 # to avoid /0
        if lensed:
            C_EE = self.C_EE_lensed(norm_ls)
            C_BB = self.C_BB_lensed(norm_ls)
        else:
            C_EE = self.C_EE(norm_ls)
            C_BB = np.zeros(norm_ls.shape)
        C_EE_n = self.C_EE_n(norm_ls)
        C_BB_n = self.C_BB_n(norm_ls)
        phi_l = getAngles(ls)
        cos2phi = np.cos(2*phi_l)
        sin2phi = np.sin(2*phi_l)
        if null:
            if lensed:
                E = self.E_lensed.fourier + self.E_noise.fourier
                B = self.B_lensed.fourier + self.B_noise.fourier
            else:
                E = self.E_prim.fourier + self.E_noise.fourier
                B = self.B_prim.fourier + self.B_noise.fourier
        else:
            E = self.E_rot.fourier + self.E_noise.fourier
            B = self.B_rot.fourier + self.B_noise.fourier
        ones = np.ones((self.N, self.N//2+1))

        qe = [(E, B)]
        f_factor1 = [(2 * C_EE, ones), (ones, -2 * C_BB)]
        f_factor2 = [(cos2phi, cos2phi), (sin2phi, sin2phi)]
        denom = [(1 / (C_EE + C_EE_n), 1 / (C_BB + C_BB_n))]
        convert = [(ones * (self.N/self.W)**2, ones * (self.N/self.W)**2)]

        factors_est = expandProd(qe, f_factor1, f_factor2, denom)
        factors_norm = expandProd(f_factor1, f_factor2, f_factor1, f_factor2, denom, convert)

        return factors_est, factors_norm

    def rotation_TB(self, null=False):
        ls = self.ls_grid.copy()
        norm_ls = self.norm_ls_grid.copy()
        norm_ls[0,0] = 1e-10 # to avoid /0
        C_TT = self.C_TT_lensed(norm_ls)
        C_TE = self.C_TE_lensed(norm_ls)
        C_BB = self.C_BB_lensed(norm_ls)
        C_TT_n = self.C_TT_n(norm_ls)
        C_BB_n = self.C_BB_n(norm_ls)
        phi_l = getAngles(ls)
        cos2phi = np.cos(2*phi_l)
        sin2phi = np.sin(2*phi_l)
        if null:
            T = self.T_lensed.fourier + self.T_noise.fourier
            B = self.B_lensed.fourier + self.B_noise.fourier
        else:
            T = self.T_rot.fourier + self.T_noise.fourier
            B = self.B_rot.fourier + self.B_noise.fourier
        ones = np.ones((self.N, self.N//2+1))

        qe = [(T, B)]
        f_factor1 = [(2 * C_TE, ones)]
        f_factor2 = [(cos2phi, cos2phi), (sin2phi, sin2phi)]
        denom = [(1 / (C_TT + C_TT_n), 1 / (C_BB + C_BB_n))]
        convert = [(ones * (self.N/self.W)**2, ones * (self.N/self.W)**2)]

        factors_est = expandProd(qe, f_factor1, f_factor2, denom)
        factors_norm = expandProd(f_factor1, f_factor2, f_factor1, f_factor2, denom, convert)

        return factors_est, factors_norm

    def rotation_TE(self, null=False, lensed=True):
        '''
        null: True if there's no rotation in the input maps, False if there is
        lensed: whether the input maps are lensed
        '''
        ls = self.ls_grid.copy()
        norm_ls = self.norm_ls_grid.copy()
        norm_ls[0,0] = 1e-10 # to avoid /0
        if lensed:
            C_TT = self.C_TT_lensed(norm_ls)
            C_EE = self.C_EE_lensed(norm_ls)
            C_TE = self.C_TE_lensed(norm_ls)
        else:
            C_TT = self.C_TT(norm_ls)
            C_EE = self.C_EE(norm_ls)
            C_TE = self.C_TE_lensed(norm_ls)
        C_TT_n = self.C_TT_n(norm_ls)
        C_EE_n = self.C_EE_n(norm_ls)
        phi_l = getAngles(ls)
        cos2phi = np.cos(2*phi_l)
        sin2phi = np.sin(2*phi_l)
        if null:
            if lensed:
                T = self.T_lensed.fourier + self.T_noise.fourier
                E = self.E_lensed.fourier + self.E_noise.fourier
            else:
                T = self.T_prim.fourier + self.T_noise.fourier
                E = self.E_prim.fourier + self.E_noise.fourier
        else:
            T = self.T_rot.fourier + self.T_noise.fourier
            E = self.E_rot.fourier + self.E_noise.fourier
        ones = np.ones((self.N, self.N//2+1))

        qe = [(T, E)]
        f_factor1 = [(-2 * C_TE, ones)]
        f_factor2 = [(cos2phi, cos2phi), (sin2phi, sin2phi)]
        denom = [(1 / (C_TT + C_TT_n), 1 / (C_EE + C_EE_n))]
        convert = [(ones * (self.N/self.W)**2, ones * (self.N/self.W)**2)]

        factors_est = expandProd(qe, f_factor1, f_factor2, denom)
        factors_norm = expandProd(f_factor1, f_factor2, f_factor1, f_factor2, denom, convert)

        return factors_est, factors_norm