import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_text_diagram_does_not_depend_on_insertion_order():
    q = cirq.LineQubit.range(4)
    ops = [cirq.CNOT(q[0], q[3]), cirq.CNOT(q[1], q[2])]
    m1, m2 = (cirq.Moment(ops), cirq.Moment(ops[::-1]))
    assert m1 == m2
    assert str(m1) == str(m2)