from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
def redistribute_forces_optimization(self, forces):
    """Redistribute forces within a triple when performing structure
        optimizations.

        The redistributed forces needs to be further adjusted using the
        appropriate Lagrange multipliers as implemented in adjust_forces."""
    forces_n, forces_m, forces_o = self.get_slices(forces)
    C1_1 = self.C1[:, 0, None]
    C1_2 = self.C1[:, 1, None]
    C4_1 = self.C4[:, 0, None]
    C4_2 = self.C4[:, 1, None]
    fr_n = (1 - C4_1 * C1_1) * forces_n - C4_1 * (C1_2 * forces_m - forces_o)
    fr_m = (1 - C4_2 * C1_2) * forces_m - C4_2 * (C1_1 * forces_n - forces_o)
    fr_o = (1 - 1 / (C1_1 ** 2 + C1_2 ** 2 + 1)) * forces_o + C4_1 * forces_n + C4_2 * forces_m
    return (fr_n, fr_m, fr_o)