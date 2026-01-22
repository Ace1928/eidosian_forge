from __future__ import annotations
import copy
import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import cos, sin
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import scipy.cluster
import spglib
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule, PeriodicSite, Structure
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list, pbc_diff
from pymatgen.util.due import Doi, due
def get_point_group_operations(self, cartesian=False):
    """Return symmetry operations as a list of SymmOp objects. By default returns
        fractional coord symm ops. But Cartesian can be returned too.

        Args:
            cartesian (bool): Whether to return SymmOps as Cartesian or
                direct coordinate operations.

        Returns:
            list[SymmOp]: Point group symmetry operations.
        """
    rotation, _translation = self._get_symmetry()
    symm_ops = []
    seen = set()
    mat = self._structure.lattice.matrix.T
    inv_mat = self._structure.lattice.inv_matrix.T
    for rot in rotation:
        rot_hash = rot.tobytes()
        if rot_hash in seen:
            continue
        seen.add(rot_hash)
        if cartesian:
            rot = np.dot(mat, np.dot(rot, inv_mat))
        op = SymmOp.from_rotation_and_translation(rot, np.array([0, 0, 0]))
        symm_ops.append(op)
    return symm_ops