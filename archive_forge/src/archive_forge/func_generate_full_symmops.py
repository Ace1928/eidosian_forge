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
def generate_full_symmops(symmops: Sequence[SymmOp], tol: float) -> Sequence[SymmOp]:
    """Recursive algorithm to permute through all possible combinations of the initially
    supplied symmetry operations to arrive at a complete set of operations mapping a
    single atom to all other equivalent atoms in the point group. This assumes that the
    initial number already uniquely identifies all operations.

    Args:
        symmops (list[SymmOp]): Initial set of symmetry operations.
        tol (float): Tolerance for detecting symmetry.

    Returns:
        list[SymmOp]: Full set of symmetry operations.
    """
    UNIT = np.eye(4)
    generators = [op.affine_matrix for op in symmops if not np.allclose(op.affine_matrix, UNIT)]
    if not generators:
        return symmops
    full = list(generators)
    for g in full:
        for s in generators:
            op = np.dot(g, s)
            d = np.abs(full - op) < tol
            if not np.any(np.all(np.all(d, axis=2), axis=1)):
                full.append(op)
            if len(full) > 1000:
                warnings.warn(f'{len(full)} matrices have been generated. The tol may be too small. Please terminate and rerun with a different tolerance.')
    d = np.abs(full - UNIT) < tol
    if not np.any(np.all(np.all(d, axis=2), axis=1)):
        full.append(UNIT)
    return [SymmOp(op) for op in full]