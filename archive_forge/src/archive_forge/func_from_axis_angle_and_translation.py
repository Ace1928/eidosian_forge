from __future__ import annotations
import re
import string
import typing
import warnings
from math import cos, pi, sin, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.due import Doi, due
from pymatgen.util.string import transformation_to_string
@staticmethod
def from_axis_angle_and_translation(axis: ArrayLike, angle: float, angle_in_radians: bool=False, translation_vec: ArrayLike=(0, 0, 0)) -> SymmOp:
    """Generates a SymmOp for a rotation about a given axis plus translation.

        Args:
            axis: The axis of rotation in Cartesian space. For example,
                [1, 0, 0]indicates rotation about x-axis.
            angle (float): Angle of rotation.
            angle_in_radians (bool): Set to True if angles are given in
                radians. Or else, units of degrees are assumed.
            translation_vec: A translation vector. Defaults to zero.

        Returns:
            SymmOp for a rotation about given axis and translation.
        """
    if isinstance(axis, (tuple, list)):
        axis = np.array(axis)
    vec = np.array(translation_vec)
    ang = angle if angle_in_radians else angle * pi / 180
    cos_a = cos(ang)
    sin_a = sin(ang)
    unit_vec = axis / np.linalg.norm(axis)
    rot_mat = np.zeros((3, 3))
    rot_mat[0, 0] = cos_a + unit_vec[0] ** 2 * (1 - cos_a)
    rot_mat[0, 1] = unit_vec[0] * unit_vec[1] * (1 - cos_a) - unit_vec[2] * sin_a
    rot_mat[0, 2] = unit_vec[0] * unit_vec[2] * (1 - cos_a) + unit_vec[1] * sin_a
    rot_mat[1, 0] = unit_vec[0] * unit_vec[1] * (1 - cos_a) + unit_vec[2] * sin_a
    rot_mat[1, 1] = cos_a + unit_vec[1] ** 2 * (1 - cos_a)
    rot_mat[1, 2] = unit_vec[1] * unit_vec[2] * (1 - cos_a) - unit_vec[0] * sin_a
    rot_mat[2, 0] = unit_vec[0] * unit_vec[2] * (1 - cos_a) - unit_vec[1] * sin_a
    rot_mat[2, 1] = unit_vec[1] * unit_vec[2] * (1 - cos_a) + unit_vec[0] * sin_a
    rot_mat[2, 2] = cos_a + unit_vec[2] ** 2 * (1 - cos_a)
    return SymmOp.from_rotation_and_translation(rot_mat, vec)