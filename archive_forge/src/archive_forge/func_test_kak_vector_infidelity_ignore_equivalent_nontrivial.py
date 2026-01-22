import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
def test_kak_vector_infidelity_ignore_equivalent_nontrivial():
    x, y, z = (np.pi / 4, 1, 0.5)
    kak_0 = cirq.kak_canonicalize_vector(x, y, z).interaction_coefficients
    kak_1 = cirq.kak_canonicalize_vector(x - 0.001, y, z).interaction_coefficients
    inf_check_equivalent = kak_vector_infidelity(kak_0, kak_1, False)
    inf_ignore_equivalent = kak_vector_infidelity(kak_0, kak_1, True)
    assert inf_check_equivalent < inf_ignore_equivalent