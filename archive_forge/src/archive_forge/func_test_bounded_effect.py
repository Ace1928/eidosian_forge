import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_bounded_effect():
    q = cirq.NamedQubit('q')
    op0 = cirq.GateOperation(cirq.testing.SingleQubitGate(), [q])
    assert cirq.trace_distance_bound(op0) >= 1
    op1 = cirq.GateOperation(cirq.Z ** 1e-06, [q])
    op1_bound = cirq.trace_distance_bound(op1)
    assert op1_bound == cirq.trace_distance_bound(cirq.Z ** 1e-06)