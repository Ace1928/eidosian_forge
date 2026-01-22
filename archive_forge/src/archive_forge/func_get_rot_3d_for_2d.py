from __future__ import annotations
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import polar
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, fast_norm
from pymatgen.core.interface import Interface, label_termination
from pymatgen.core.surface import SlabGenerator
def get_rot_3d_for_2d(film_matrix, sub_matrix) -> np.ndarray:
    """Find transformation matrix that will rotate and strain the film to the substrate while preserving the c-axis."""
    film_matrix = np.array(film_matrix)
    film_matrix = film_matrix.tolist()[:2]
    film_matrix.append(np.cross(film_matrix[0], film_matrix[1]))
    sub_matrix = np.array(sub_matrix)
    sub_matrix = sub_matrix.tolist()[:2]
    temp_sub = np.cross(sub_matrix[0], sub_matrix[1]).astype(float)
    temp_sub = temp_sub * fast_norm(np.array(film_matrix[2], dtype=float))
    sub_matrix.append(temp_sub)
    transform_matrix = np.transpose(np.linalg.solve(film_matrix, sub_matrix))
    rot, _ = polar(transform_matrix)
    return rot