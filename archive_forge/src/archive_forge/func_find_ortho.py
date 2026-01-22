from ase.lattice.bravais import Bravais, reduceindex
import numpy as np
from ase.data import reference_states as _refstate
def find_ortho(self, idx):
    """Replace keyword 'ortho' or 'orthogonal' with a direction."""
    for i in range(3):
        if isinstance(idx[i], str) and (idx[i].lower() == 'ortho' or idx[i].lower() == 'orthogonal'):
            if self.debug:
                print('Calculating orthogonal direction', i)
                print(idx[i - 2], 'X', idx[i - 1], end=' ')
            idx[i] = reduceindex(np.cross(idx[i - 2], idx[i - 1]))
            if self.debug:
                print('=', idx[i])