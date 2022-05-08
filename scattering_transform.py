from cmb_simulation import *
from kymatio.numpy import Scattering2D
from tqdm import tqdm
from time import time

class Coefs:
    '''
    Class for storing and accessing reduced scattering coef-like triangular list-of-list
    '''
    def __init__(self, J, arr, flat=False):
        self.J = J
        self.val = self.unflatten(arr) if flat else arr

    def __getitem__(self, key):
        if type(key) is tuple:
            j1, j2 = key
            if j1>=0 and j1<self.J-1 and j2>j1 and j2<self.J:
                return self.val[j1][j2-j1-1]
            else:
                raise KeyError()
        else:
            raise KeyError()

    def flatten(self):
        return np.concatenate(self.val)

    def unflatten(self, arr):
        if len(arr) != self.J * (self.J-1) // 2:
            raise ValueError('Length of arr must be J*(J-1)/2')
        val = []
        start = 0
        for j1 in range(self.J-1):
            length = self.J-j1-1
            val.append(list(arr[start:start+length]))
            start += length
        return val

    def plot(self, ylabel='', logaxis=False, **kwargs):
        for j1 in range(self.J-1):
            j2s = np.arange(j1+1, self.J)
            fmt = kwargs.pop('fmt') if 'fmt' in kwargs else ''
            plt.plot(j2s, self.val[j1], fmt, **kwargs)
        if logaxis:
            plt.yscale('log')
        plt.xlabel('$j_2$')
        plt.ylabel(ylabel)

def reduceCoefs(J, Sx):
    '''
    Sx : list of dicts
        Output of Scattering2D in list mode, applied to a real map
    '''
    s1 = [np.mean(
        [sx['coef'] for sx in Sx if sx['j'] == (j,)]
    ) for j in range(J-1)]
    s2 = [
        [np.mean(
            [sx['coef'] for sx in Sx if sx['j'] == (j1, j2)]
        ) for j2 in range(j1+1, J)] for j1 in range(J-1)
    ]
    return s1, s2

class ReducedCoefs(Coefs):
    '''
    Class storing reduced coefs, i.e. scattering coefs averaged over position and orientation
    '''
    def __init__(self, J, s1, s2):
        self.J = J
        self.s1 = s1
        self.s2 = s2
        self.val = self.s2

    def __getitem__(self, key):
        if type(key) is int and key>=0 and key<self.J-1:
            return self.s1[key]
        else:
            return super().__getitem__(key)

    def logReducedCoefs(self):
        log_s1 = np.log(self.s1)
        val = [
            [
                np.log(self[j1,j2]/self[j1]) for j2 in range(j1+1, self.J)
            ] for j1 in range(self.J-1)
        ]
        return ReducedCoefs(self.J, log_s1, val)

    def flatten(self, include_s1=False):
        if include_s1:
            return np.concatenate((self.s1, super().flatten()))
        else:
            return super().flatten()

class CoefStats:
    def __init__(self, J, init=[], **params):
        '''
        init : list of Coefs
        '''
        self.J = J
        self.params = params
        self.coefs = list(init)

    def append(self, *coefs):
        '''
        coefs : Coefs
        '''
        self.coefs += coefs

    def cov(self, include_s1=False):
        data = np.array([c.flatten(include_s1=include_s1) for c in self.coefs])
        return np.cov(data.T)

    def mean(self, include_s1=False):
        data = np.array([c.flatten(include_s1=include_s1) for c in self.coefs])
        return Coefs(self.J, np.mean(data, axis=0), flat=True)

    def std(self, include_s1=False):
        data = np.array([c.flatten(include_s1=include_s1) for c in self.coefs])
        return Coefs(self.J, np.std(data, axis=0), flat=True)

    def plot(self, ylabel='', logaxis=False, **kwargs):
        mean = self.mean()
        std = self.std()
        for j1 in range(self.J-1):
            j2s = np.arange(j1+1, self.J)
            plt.errorbar(j2s, mean.val[j1], std.val[j1], **kwargs)
        if logaxis:
            plt.yscale('log')
        plt.xlabel('$j_2$')
        plt.ylabel(ylabel)

class STQueue:
    def __init__(self, J, N):
        self.J = J
        self.N = N
        self.S = Scattering2D(J, (N, N), out_type='list')

    def init(self, newmap_fun, params):
        '''
        newmap_fun : dict -> (N, N) array
            Function called to generate a realisation of a real map for a set of parameters given as a dict

        params : list of dict
            List of parameters to use to generate realisations
            Each dict is {'multi': int, 'A': float, ...}, where 'multi' is the number of repeat realisations for this set of parameters
        '''
        self.newMap = newmap_fun
        self.coef_stats = [CoefStats(self.J, **{k: v for k, v in p.items() if k != 'multi'}) for p in params]
        multis = np.array([p['multi'] for p in params])
        self.params = params
        self.queue = [] # queue of parameter sets to run in order
        while (multis > 0).any():
            next_jobs = [i for i in range(len(params)) if multis[i] > 0]
            self.queue += next_jobs
            multis = np.array([max(m-1, 0) for m in multis]) # decrement non-zero entries
        self.i = 0 # index of queue to be run next

    def scatteringTransform(self, **kwargs):
        '''
        Scattering transform a single realisation, returned and not saved

        kwargs
            passed to newMap to generate realisation
        '''
        X = self.newMap(**kwargs)
        s1, s2 = reduceCoefs(self.J, self.S(X))
        return ReducedCoefs(self.J, s1, s2)

    def run(self):
        time1 = -1
        time2 = time()
        while self.i < len(self.queue):
            m = self.params[self.queue[self.i]]['multi']
            p = {k: v for k, v in self.params[self.queue[self.i]].items() if k != 'multi'}
            current = len(self.coef_stats[self.queue[self.i]].coefs)
            subst = (current+1, m, self.i+1, len(self.queue), str(p))
            duration = time2 - time1
            duration_str = ', previous cycle took %.2f s' % duration if time1 > 0 else ''
            print('Simulating realisation %d/%d (%d/%d in queue) with parameters %s' % subst + duration_str, end='\r')
            try:
                coef = self.scatteringTransform(**p).logReducedCoefs()
            except KeyboardInterrupt:
                print('\nInterrupted')
                return
            self.coef_stats[self.queue[self.i]].append(coef)
            self.i += 1
            time1 = time2
            time2 = time()
        print('%d realisations simulated' % len(self.queue))