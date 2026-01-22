import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
@pytest.mark.parametrize('circuit, expected_superoperator', ((cirq.Circuit(cirq.I(q0)), np.eye(4)), (cirq.Circuit(cirq.IdentityGate(2).on(q0, q1)), np.eye(16)), (cirq.Circuit(cirq.H(q0)), np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2), (cirq.Circuit(cirq.S(q0)), np.diag([1, -1j, 1j, 1])), (cirq.Circuit(cirq.depolarize(0.75).on(q0)), np.outer([1, 0, 0, 1], [1, 0, 0, 1]) / 2), (cirq.Circuit(cirq.X(q0), cirq.depolarize(0.75).on(q0)), np.outer([1, 0, 0, 1], [1, 0, 0, 1]) / 2), (cirq.Circuit(cirq.Y(q0), cirq.depolarize(0.75).on(q0)), np.outer([1, 0, 0, 1], [1, 0, 0, 1]) / 2), (cirq.Circuit(cirq.Z(q0), cirq.depolarize(0.75).on(q0)), np.outer([1, 0, 0, 1], [1, 0, 0, 1]) / 2), (cirq.Circuit(cirq.H(q0), cirq.depolarize(0.75).on(q0)), np.outer([1, 0, 0, 1], [1, 0, 0, 1]) / 2), (cirq.Circuit(cirq.H(q0), cirq.H(q0)), np.eye(4)), (cirq.Circuit(cirq.H(q0), cirq.CNOT(q1, q0), cirq.H(q0)), np.diag([1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1]))))
def test_circuit_superoperator_fixed_values(circuit, expected_superoperator):
    """Tests Circuit._superoperator_() on a few simple circuits."""
    assert circuit._has_superoperator_()
    assert np.allclose(circuit._superoperator_(), expected_superoperator)