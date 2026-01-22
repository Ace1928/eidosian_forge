from __future__ import annotations
import itertools
import math
import warnings
from typing import TYPE_CHECKING, Literal
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import root
from scipy.special import factorial
from pymatgen.analysis.elasticity.strain import Strain
from pymatgen.analysis.elasticity.stress import Stress
from pymatgen.core.tensors import DEFAULT_QUAD, SquareTensor, Tensor, TensorCollection, get_uvec
from pymatgen.core.units import Unit
from pymatgen.util.due import Doi, due
def get_symbol_list(rank, dim=6):
    """
    Returns a symbolic representation of the Voigt-notation
    tensor that places identical symbols for entries related
    by index transposition, i. e. C_1121 = C_1211 etc.

    Args:
        dim (int): dimension of matrix/tensor, e. g. 6 for
            voigt notation and 3 for standard
        rank (int): rank of tensor, e. g. 3 for third-order ECs

    Returns:
        c_vec (array): array representing distinct indices
        c_arr (array): array representing tensor with equivalent
            indices assigned as above
    """
    indices = list(itertools.combinations_with_replacement(range(dim), r=rank))
    c_vec = np.zeros(len(indices), dtype=object)
    c_arr = np.zeros([dim] * rank, dtype=object)
    for n, idx in enumerate(indices):
        c_vec[n] = sp.Symbol('c_' + ''.join(map(str, idx)))
        for perm in itertools.permutations(idx):
            c_arr[perm] = c_vec[n]
    return (c_vec, c_arr)