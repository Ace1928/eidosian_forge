import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_gate_on_operation_besides_gate_operation():
    a, b = cirq.LineQubit.range(2)
    op = -1j * cirq.X(a) * cirq.Y(b)
    assert isinstance(op.gate, cirq.DensePauliString)
    assert op.gate == -1j * cirq.DensePauliString('XY')
    assert not isinstance(op.gate, cirq.XPowGate)