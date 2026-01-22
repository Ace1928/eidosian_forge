import numpy as np
from ase.cluster.factory import ClusterFactory
from ase.data import reference_states as _refstate
def set_surfaces_layers(self, surfaces, layers):
    for i, s in enumerate(surfaces):
        if len(s) == 4:
            a, b, c, d = s
            if a + b + c != 0:
                raise ValueError(('(%d,%d,%d,%d) is not a valid hexagonal Miller ' + 'index, as the sum of the first three numbers ' + 'should be zero.') % (a, b, c, d))
            surfaces[i] = [a, b, d]
    ClusterFactory.set_surfaces_layers(self, surfaces, layers)