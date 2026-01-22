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
@staticmethod
def get_voigt_dict(rank):
    """Returns a dictionary that maps indices in the tensor to those
        in a voigt representation based on input rank.

        Args:
            rank (int): Tensor rank to generate the voigt map
        """
    vdict = {}
    for ind in itertools.product(*[range(3)] * rank):
        v_ind = ind[:rank % 2]
        for j in range(rank // 2):
            pos = rank % 2 + 2 * j
            v_ind += (reverse_voigt_map[ind[pos:pos + 2]],)
        vdict[ind] = v_ind
    return vdict