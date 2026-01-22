from __future__ import annotations
import collections
import itertools
import os
import string
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from monty.serialization import loadfn
from scipy.linalg import polar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def einsum_sequence(self, other_arrays, einsum_string=None):
    """Calculates the result of an einstein summation expression."""
    if not isinstance(other_arrays, list):
        raise ValueError('other tensors must be list of tensors or tensor input')
    other_arrays = [np.array(a) for a in other_arrays]
    if not einsum_string:
        lc = string.ascii_lowercase
        einsum_string = lc[:self.rank]
        other_ranks = [len(a.shape) for a in other_arrays]
        idx = self.rank - sum(other_ranks)
        for length in other_ranks:
            einsum_string += ',' + lc[idx:idx + length]
            idx += length
    einsum_args = [self, *other_arrays]
    return np.einsum(einsum_string, *einsum_args)