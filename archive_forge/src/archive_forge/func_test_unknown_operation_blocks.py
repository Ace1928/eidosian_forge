import dataclasses
import pytest
import numpy as np
import sympy
import cirq
from cirq.transformers.eject_z import _is_swaplike
def test_unknown_operation_blocks():
    q = cirq.NamedQubit('q')

    class UnknownOp(cirq.Operation):

        @property
        def qubits(self):
            return [q]

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()
    u = UnknownOp()
    assert_optimizes(before=cirq.Circuit([cirq.Moment([cirq.Z(q)]), cirq.Moment([u])]), expected=cirq.Circuit([cirq.Moment([cirq.Z(q)]), cirq.Moment([u])]))