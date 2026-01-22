import sys
import numpy as np
from itertools import combinations_with_replacement
import ase.units as u
from ase.parallel import parprint, paropen
from ase.vibrations.resonant_raman import ResonantRaman
from ase.vibrations.franck_condon import FranckCondonOverlap
from ase.vibrations.franck_condon import FranckCondonRecursive
def calculate_energies_and_modes(self):
    if hasattr(self, 'im_r'):
        return
    ResonantRaman.calculate_energies_and_modes(self)
    om_Q = self.om_Q[self.skip:]
    om_v = om_Q
    ndof = len(om_Q)
    n_vQ = np.eye(ndof, dtype=int)
    l_Q = range(ndof)
    ind_v = list(combinations_with_replacement(l_Q, 1))
    if self.combinations > 1:
        if not self.combinations == 2:
            raise NotImplementedError
        for c in range(2, self.combinations + 1):
            ind_v += list(combinations_with_replacement(l_Q, c))
        nv = len(ind_v)
        n_vQ = np.zeros((nv, ndof), dtype=int)
        om_v = np.zeros(nv, dtype=float)
        for j, wt in enumerate(ind_v):
            for i in wt:
                n_vQ[j, i] += 1
        om_v = n_vQ.dot(om_Q)
    self.ind_v = ind_v
    self.om_v = om_v
    self.n_vQ = n_vQ
    self.d_vQ = np.where(n_vQ > 0, 1, 0)