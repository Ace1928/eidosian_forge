from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def celldiff(cell1, cell2):
    """Return a unitless measure of the difference between two cells."""
    cell1 = Cell.ascell(cell1).complete()
    cell2 = Cell.ascell(cell2).complete()
    v1v2 = cell1.volume * cell2.volume
    if v1v2 == 0:
        raise ZeroDivisionError('Cell volumes are zero')
    scale = v1v2 ** (-1.0 / 3.0)
    x1 = cell1 @ cell1.T
    x2 = cell2 @ cell2.T
    dev = scale * np.abs(x2 - x1).max()
    return dev