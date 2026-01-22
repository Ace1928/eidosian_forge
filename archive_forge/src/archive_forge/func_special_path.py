from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@property
def special_path(self) -> str:
    """Get default special k-point path for this lattice as a string.

        >>> BCT(3, 5).special_path
        'GXYSGZS1NPY1Z,XP'
        """
    return self._variant.special_path