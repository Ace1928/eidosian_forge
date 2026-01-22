import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_gate_on():
    q = cirq.LineQubit(0)
    g1 = cirq.PauliStringPhasorGate(cirq.DensePauliString('X', coefficient=-1), exponent_neg=0.25, exponent_pos=-0.5)
    op1 = g1.on(q)
    assert isinstance(op1, cirq.PauliStringPhasor)
    assert op1.qubits == (q,)
    assert op1.gate == g1
    assert op1.pauli_string == dps_x.on(q)
    assert op1.exponent_neg == -0.5
    assert op1.exponent_pos == 0.25
    g2 = cirq.PauliStringPhasorGate(dps_x, exponent_neg=0.75, exponent_pos=-0.125)
    op2 = g2.on(q)
    assert isinstance(op2, cirq.PauliStringPhasor)
    assert op2.qubits == (q,)
    assert op2.gate == g2
    assert op2.pauli_string == dps_x.on(q)
    assert op2.exponent_neg == 0.75
    assert op2.exponent_pos == -0.125