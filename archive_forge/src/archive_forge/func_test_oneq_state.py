import numpy as np
import pytest
import cirq
def test_oneq_state():
    q0, q1 = cirq.LineQubit.range(2)
    st0 = cirq.KET_PLUS(q0)
    assert str(st0) == '+X(q(0))'
    st1 = cirq.KET_PLUS(q1)
    assert st0 != st1
    assert st0 == cirq.KET_PLUS.on(q0)