import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_dense_pauli_string():
    gate = cirq.SingleQubitCliffordGate.from_xz_map((cirq.X, True), (cirq.Y, False))
    assert gate.dense_pauli_string(cirq.X) == cirq.DensePauliString('X', coefficient=-1)
    assert gate.dense_pauli_string(cirq.Z) == cirq.DensePauliString('Y')