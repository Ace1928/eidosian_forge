import numpy as np
import pytest
import cirq
def test_product_iter():
    q0, q1, q2 = cirq.LineQubit.range(3)
    ps = cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1) * cirq.KET_ZERO(q2)
    should_be = [(q0, cirq.KET_PLUS), (q1, cirq.KET_PLUS), (q2, cirq.KET_ZERO)]
    assert list(ps) == should_be
    assert len(ps) == 3