import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_op_has_no_kraus():

    class EmptyGate(cirq.testing.SingleQubitGate):
        pass
    m = cirq.Moment(EmptyGate().on(cirq.NamedQubit('a')))
    assert not cirq.has_kraus(m)
    assert not m._has_superoperator_()
    assert m._kraus_() is NotImplemented
    assert m._superoperator_() is NotImplemented
    assert cirq.kraus(m, default=None) is None