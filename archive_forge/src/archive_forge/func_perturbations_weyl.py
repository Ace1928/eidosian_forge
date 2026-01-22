import itertools
import numpy as np
import pytest
import cirq
import sympy
def perturbations_weyl(x, y, z, amount=1e-10):
    return perturbations_gate(cirq.KakDecomposition(interaction_coefficients=(x, y, z)), amount)