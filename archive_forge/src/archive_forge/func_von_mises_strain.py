from __future__ import annotations
import collections
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
import scipy
from pymatgen.core.lattice import Lattice
from pymatgen.core.tensors import SquareTensor, symmetry_reduce
@property
def von_mises_strain(self):
    """Equivalent strain to Von Mises Stress."""
    eps = self - 1 / 3 * np.trace(self) * np.identity(3)
    return np.sqrt(np.sum(eps * eps) * 2 / 3)