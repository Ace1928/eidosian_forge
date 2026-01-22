import numpy as np
from itertools import combinations_with_replacement
from math import erf
from scipy.spatial.distance import cdist
from ase.neighborlist import NeighborList
from ase.utils import pbc2pbc
def plot_fingerprints(self, a, prefix=''):
    """ Function for quickly plotting all the fingerprints.
        Prefix = a prefix you want to give to the resulting PNG file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        Warning("Matplotlib could not be loaded - plotting won't work")
        raise
    if 'fingerprints' in a.info and (not self.recalculate):
        fp, typedic = a.info['fingerprints']
        fp, typedic = self._json_decode(fp, typedic)
    else:
        a_top = a[-self.n_top:]
        fp, typedic = self._take_fingerprints(a_top)
        a.info['fingerprints'] = self._json_encode(fp, typedic)
    npts = int(np.ceil(self.rcut * 1.0 / self.binwidth))
    x = np.linspace(0, self.rcut, npts, endpoint=False)
    for key, val in fp.items():
        plt.plot(x, val)
        suffix = '_fp_{0}_{1}.png'.format(key[0], key[1])
        plt.savefig(prefix + suffix)
        plt.clf()