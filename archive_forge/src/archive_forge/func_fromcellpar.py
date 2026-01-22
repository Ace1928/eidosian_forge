import ase
from typing import Mapping, Sequence, Union
import numpy as np
from ase.utils.arraywrapper import arraylike
from ase.utils import pbc2pbc
@classmethod
def fromcellpar(cls, cellpar, ab_normal=(0, 0, 1), a_direction=None):
    """Return new Cell from cell lengths and angles.

        See also :func:`~ase.geometry.cell.cellpar_to_cell()`."""
    from ase.geometry.cell import cellpar_to_cell
    cell = cellpar_to_cell(cellpar, ab_normal, a_direction)
    return cls(cell)