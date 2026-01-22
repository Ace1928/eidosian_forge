import ase
from typing import Mapping, Sequence, Union
import numpy as np
from ase.utils.arraywrapper import arraylike
from ase.utils import pbc2pbc
def scaled_positions(self, positions) -> np.ndarray:
    """Calculate scaled positions from Cartesian positions.

        The scaled positions are the positions given in the basis
        of the cell vectors.  For the purpose of defining the basis, cell
        vectors that are zero will be replaced by unit vectors as per
        :meth:`~ase.cell.Cell.complete`."""
    return np.linalg.solve(self.complete().T, np.transpose(positions)).T