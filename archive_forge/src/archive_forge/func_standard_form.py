import ase
from typing import Mapping, Sequence, Union
import numpy as np
from ase.utils.arraywrapper import arraylike
from ase.utils import pbc2pbc
def standard_form(self):
    """Rotate axes such that unit cell is lower triangular. The cell
        handedness is preserved.

        A lower-triangular cell with positive diagonal entries is a canonical
        (i.e. unique) description. For a left-handed cell the diagonal entries
        are negative.

        Returns:

        rcell: the standardized cell object

        Q: ndarray
            The orthogonal transformation.  Here, rcell @ Q = cell, where cell
            is the input cell and rcell is the lower triangular (output) cell.
        """
    sign = np.sign(np.linalg.det(self))
    if sign == 0:
        sign = 1
    Q, L = np.linalg.qr(self.T)
    Q = Q.T
    L = L.T
    signs = np.sign(np.diag(L))
    indices = np.where(signs == 0)[0]
    signs[indices] = 1
    indices = np.where(signs != sign)[0]
    L[:, indices] *= -1
    Q[indices] *= -1
    return (Cell(L), Q)