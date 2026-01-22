import numpy as np
from autograd.extend import VSpace
from autograd.builtins import NamedTupleVSpace
def standard_basis(self):
    for idxs in np.ndindex(*self.shape):
        for v in [1.0, 1j]:
            vect = np.zeros(self.shape, dtype=self.dtype)
            vect[idxs] = v
            yield vect