import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_condition_removal():
    q0 = cirq.LineQubit(0)
    op = cirq.X(q0).with_tags('t1').with_classical_controls('a').with_tags('t2').with_classical_controls('b')
    op = op.without_classical_controls()
    assert not cirq.control_keys(op)
    assert not op.classical_controls
    assert not op.tags