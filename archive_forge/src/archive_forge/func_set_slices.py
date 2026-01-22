from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
def set_slices(self, a_n, a_m, a_o, a):
    a[self.n_ind] = a_n
    a[self.m_ind] = a_m
    a[self.o_ind] = a_o