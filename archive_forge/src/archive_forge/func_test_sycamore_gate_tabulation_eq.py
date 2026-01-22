import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.heuristic_decompositions.two_qubit_gate_tabulation import (
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
from cirq.testing import random_special_unitary, assert_equivalent_repr
def test_sycamore_gate_tabulation_eq():
    assert sycamore_tabulation == sycamore_tabulation
    assert sycamore_tabulation != sqrt_iswap_tabulation
    assert sycamore_tabulation != 1