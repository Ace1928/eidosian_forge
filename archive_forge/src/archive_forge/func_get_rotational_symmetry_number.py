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
def get_rotational_symmetry_number(self):
    """Return the rotational symmetry number."""
    symm_ops = self.get_symmetry_operations()
    symm_number = 0
    for symm in symm_ops:
        rot = symm.rotation_matrix
        if np.abs(np.linalg.det(rot) - 1) < 0.0001:
            symm_number += 1
    return symm_number