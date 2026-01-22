import math
import numpy as np
from ase import Atoms
from ase.cluster.base import ClusterBase
def get_diameter(self, method='volume'):
    """Returns an estimate of the cluster diameter based on two different
        methods.

        method = 'volume': Returns the diameter of a sphere with the
                           same volume as the atoms. (Default)
        
        method = 'shape': Returns the averaged diameter calculated from the
                          directions given by the defined surfaces.
        """
    if method == 'shape':
        cen = self.get_positions().mean(axis=0)
        pos = self.get_positions() - cen
        d = 0.0
        for s in self.surfaces:
            n = self.miller_to_direction(s)
            r = np.dot(pos, n)
            d += r.max() - r.min()
        return d / len(self.surfaces)
    elif method == 'volume':
        V_cell = np.abs(np.linalg.det(self.lattice_basis))
        N_cell = len(self.atomic_basis)
        N = len(self)
        return 2.0 * (3.0 * N * V_cell / (4.0 * math.pi * N_cell)) ** (1.0 / 3.0)
    else:
        return 0.0