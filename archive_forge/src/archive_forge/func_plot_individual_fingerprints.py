import numpy as np
from itertools import combinations_with_replacement
from math import erf
from scipy.spatial.distance import cdist
from ase.neighborlist import NeighborList
from ase.utils import pbc2pbc
def plot_individual_fingerprints(self, a, prefix=''):
    """ Function for plotting all the individual fingerprints.
        Prefix = a prefix for the resulting PNG file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        Warning("Matplotlib could not be loaded - plotting won't work")
        raise
    if 'individual_fingerprints' in a.info and (not self.recalculate):
        fp, typedic = a.info['individual_fingerprints']
    else:
        a_top = a[-self.n_top:]
        fp, typedic = self._take_fingerprints(a_top, individual=True)
        a.info['individual_fingerprints'] = [fp, typedic]
    npts = int(np.ceil(self.rcut * 1.0 / self.binwidth))
    x = np.linspace(0, self.rcut, npts, endpoint=False)
    for key, val in fp.items():
        for key2, val2 in val.items():
            plt.plot(x, val2)
            plt.ylim([-1, 10])
            suffix = '_individual_fp_{0}_{1}.png'.format(key, key2)
            plt.savefig(prefix + suffix)
            plt.clf()