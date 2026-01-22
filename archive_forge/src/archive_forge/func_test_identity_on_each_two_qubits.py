import itertools
from typing import Any
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_identity_on_each_two_qubits():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    q0_3, q1_3 = (q0.with_dimension(3), q1.with_dimension(3))
    assert cirq.IdentityGate(2).on_each([(q0, q1)]) == [cirq.IdentityGate(2)(q0, q1)]
    assert cirq.IdentityGate(2).on_each([(q0, q1), (q2, q3)]) == [cirq.IdentityGate(2)(q0, q1), cirq.IdentityGate(2)(q2, q3)]
    assert cirq.IdentityGate(2, (3, 3)).on_each([(q0_3, q1_3)]) == [cirq.IdentityGate(2, (3, 3))(q0_3, q1_3)]
    assert cirq.IdentityGate(2).on_each((q0, q1)) == [cirq.IdentityGate(2)(q0, q1)]
    with pytest.raises(ValueError, match='Inputs to multi-qubit gates must be Sequence'):
        cirq.IdentityGate(2).on_each(q0, q1)
    with pytest.raises(ValueError, match='All values in sequence should be Qids'):
        cirq.IdentityGate(2).on_each([[(q0, q1)]])
    with pytest.raises(ValueError, match='Expected 2 qubits'):
        cirq.IdentityGate(2).on_each([(q0,)])
    with pytest.raises(ValueError, match='Expected 2 qubits'):
        cirq.IdentityGate(2).on_each([(q0, q1, q2)])