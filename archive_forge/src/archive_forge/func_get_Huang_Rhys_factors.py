from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul
import numpy as np
from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations
def get_Huang_Rhys_factors(self, forces):
    """Evaluate Huang-Rhys factors and corresponding frequencies
        from forces on atoms in the exited electronic state.
        The double harmonic approximation is used. HR factors are
        the first approximation of FC factors,
        no combinations or higher quanta (>1) exitations are considered"""
    assert forces.shape == self.shape
    H_VV = self.H
    mm05_V = self.mm05_V
    Hm_VV = mm05_V[:, None] * H_VV * mm05_V
    Fm_V = forces.flat * mm05_V
    X_V = np.linalg.solve(Hm_VV, Fm_V)
    modes_VV = self.modes
    d_V = np.dot(modes_VV, X_V)
    s = 1e-20 / kg / C / _hbar ** 2
    S_V = s * d_V ** 2 * self.energies / 2
    indices = np.where(self.frequencies <= self.minfreq)
    np.append(indices, np.where(self.frequencies >= self.maxfreq))
    S_V = np.delete(S_V, indices)
    frequencies = np.delete(self.frequencies, indices)
    return (S_V, frequencies)