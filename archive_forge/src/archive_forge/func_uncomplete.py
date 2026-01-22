import ase
from typing import Mapping, Sequence, Union
import numpy as np
from ase.utils.arraywrapper import arraylike
from ase.utils import pbc2pbc
def uncomplete(self, pbc):
    """Return new cell, zeroing cell vectors where not periodic."""
    _pbc = np.empty(3, bool)
    _pbc[:] = pbc
    cell = self.copy()
    cell[~_pbc] = 0
    return cell