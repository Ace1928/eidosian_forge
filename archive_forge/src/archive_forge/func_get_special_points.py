from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def get_special_points(self) -> Dict[str, np.ndarray]:
    """Return a dictionary of named special k-points for this lattice."""
    if self._variant.special_points is not None:
        return self._variant.special_points
    labels = self.special_point_names
    points = self.get_special_points_array()
    return dict(zip(labels, points))