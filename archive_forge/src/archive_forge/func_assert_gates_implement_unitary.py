import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
def assert_gates_implement_unitary(gates: Sequence[cirq.testing.SingleQubitGate], intended_effect: np.ndarray, atol: float):
    actual_effect = cirq.dot(*[cirq.unitary(g) for g in reversed(gates)])
    cirq.testing.assert_allclose_up_to_global_phase(actual_effect, intended_effect, atol=atol)