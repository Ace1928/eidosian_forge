import ctypes
import numpy as np
from numpy.linalg import norm
from ase.calculators.calculator import Calculator
from ase.data import (atomic_numbers as ase_atomic_numbers,
from ase.calculators.lammps import convert
from ase.geometry import wrap_positions
def is_upper_triangular(arr, atol=1e-08):
    """test for upper triangular matrix based on numpy"""
    assert len(arr.shape) == 2
    assert arr.shape[0] == arr.shape[1]
    return np.allclose(np.tril(arr, k=-1), 0.0, atol=atol) and np.all(np.diag(arr) >= 0.0)