from __future__ import annotations
import collections
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
import scipy
from pymatgen.core.lattice import Lattice
from pymatgen.core.tensors import SquareTensor, symmetry_reduce
def convert_strain_to_deformation(strain, shape: Literal['upper', 'lower', 'symmetric']):
    """
    This function converts a strain to a deformation gradient that will
    produce that strain. Supports three methods:

    Args:
        strain (3x3 array-like): strain matrix
        shape: ('upper' | 'lower' | 'symmetric'): method for determining deformation
            'upper' produces an upper triangular defo
            'lower' produces a lower triangular defo
            'symmetric' produces a symmetric defo
    """
    strain = SquareTensor(strain)
    ft_dot_f = 2 * strain + np.eye(3)
    if shape == 'upper':
        result = scipy.linalg.cholesky(ft_dot_f)
    elif shape == 'symmetric':
        result = scipy.linalg.sqrtm(ft_dot_f)
    else:
        raise ValueError('shape must be "upper" or "symmetric"')
    return Deformation(result)