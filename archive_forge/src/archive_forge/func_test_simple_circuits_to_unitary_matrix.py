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
@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_simple_circuits_to_unitary_matrix(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = circuit_cls(cirq.CNOT(a, b), cirq.Z(b), cirq.CNOT(a, b))
    assert cirq.has_unitary(c)
    m = c.unitary()
    cirq.testing.assert_allclose_up_to_global_phase(m, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]), atol=1e-08)
    for expected in [np.diag([1, 1j, -1, -1j]), cirq.unitary(cirq.CNOT)]:

        class Passthrough(cirq.testing.TwoQubitGate):

            def _unitary_(self) -> np.ndarray:
                return expected
        c = circuit_cls(Passthrough()(a, b))
        m = c.unitary()
        cirq.testing.assert_allclose_up_to_global_phase(m, expected, atol=1e-08)