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
@classmethod
def from_xyz_str(cls, xyz_str: str) -> Self:
    """
        Args:
            xyz_str: string of the form 'x, y, z', '-x, -y, z', '-2y+1/2, 3x+1/2, z-y+1/2', etc.

        Returns:
            SymmOp
        """
    rot_matrix = np.zeros((3, 3))
    trans = np.zeros(3)
    tokens = xyz_str.strip().replace(' ', '').lower().split(',')
    re_rot = re.compile('([+-]?)([\\d\\.]*)/?([\\d\\.]*)([x-z])')
    re_trans = re.compile('([+-]?)([\\d\\.]+)/?([\\d\\.]*)(?![x-z])')
    for i, tok in enumerate(tokens):
        for m in re_rot.finditer(tok):
            factor = -1.0 if m.group(1) == '-' else 1.0
            if m.group(2) != '':
                factor *= float(m.group(2)) / float(m.group(3)) if m.group(3) != '' else float(m.group(2))
            j = ord(m.group(4)) - 120
            rot_matrix[i, j] = factor
        for m in re_trans.finditer(tok):
            factor = -1 if m.group(1) == '-' else 1
            num = float(m.group(2)) / float(m.group(3)) if m.group(3) != '' else float(m.group(2))
            trans[i] = num * factor
    return cls.from_rotation_and_translation(rot_matrix, trans)