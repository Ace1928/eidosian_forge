from typing import List, Optional
import numpy as np
from ase.data import atomic_numbers as ref_atomic_numbers
from ase.spacegroup import Spacegroup
from ase.cluster.base import ClusterBase
from ase.cluster.cluster import Cluster
def set_lattice_size(self, center):
    if center is None:
        offset = np.zeros(3)
    else:
        offset = np.array(center)
        if (offset > 1.0).any() or (offset < 0.0).any():
            raise ValueError('Center offset must lie within the lattice unit                                   cell.')
    max = np.ones(3)
    min = -np.ones(3)
    v = np.linalg.inv(self.lattice_basis.T)
    for s, l in zip(self.surfaces, self.layers):
        n = self.miller_to_direction(s) * self.get_layer_distance(s, l)
        k = np.round(np.dot(v, n), 2)
        for i in range(3):
            if k[i] > 0.0:
                k[i] = np.ceil(k[i])
            elif k[i] < 0.0:
                k[i] = np.floor(k[i])
        if self.debug > 1:
            print('Spaning %i layers in %s in lattice basis ~ %s' % (l, s, k))
        max[k > max] = k[k > max]
        min[k < min] = k[k < min]
    self.center = np.dot(offset - min, self.lattice_basis)
    self.size = (max - min + np.ones(3)).astype(int)