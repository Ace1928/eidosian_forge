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
def special_point_names(self) -> List[str]:
    """Return all special point names as a list of strings.

        >>> BCT(3, 5).special_point_names
        ['G', 'N', 'P', 'S', 'S1', 'X', 'Y', 'Y1', 'Z']
        """
    labels = parse_path_string(self._variant.special_point_names)
    assert len(labels) == 1
    return labels[0]