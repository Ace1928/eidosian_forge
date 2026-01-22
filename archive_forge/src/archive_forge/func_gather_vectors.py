from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
def gather_vectors(self, pos):
    v0 = [pos[k] - pos[h] for h, k, l, m in self.indices]
    v1 = [pos[l] - pos[k] for h, k, l, m in self.indices]
    v2 = [pos[m] - pos[l] for h, k, l, m in self.indices]
    return (v0, v1, v2)