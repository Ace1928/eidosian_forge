import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
def get_linear_kpoint_axis(self, eps=1e-05):
    """Define x axis suitable for plotting a band structure.

        See :func:`ase.dft.kpoints.labels_from_kpts`."""
    index2name = self._find_special_point_indices(eps)
    indices = sorted(index2name)
    labels = [index2name[index] for index in indices]
    xcoords, special_xcoords = indices_to_axis_coords(indices, self.kpts, self.cell)
    return (xcoords, special_xcoords, labels)