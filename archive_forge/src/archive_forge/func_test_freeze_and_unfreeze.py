import pytest
import sympy
import cirq
def test_freeze_and_unfreeze():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(a), cirq.H(b))
    f = c.freeze()
    assert f == c
    assert cirq.approx_eq(f, c)
    ff = f.freeze()
    assert ff is f
    unf = f.unfreeze()
    assert unf.moments == c.moments
    assert unf is not c
    cc = c.unfreeze()
    assert cc is not c
    fcc = cc.freeze()
    assert fcc.moments == f.moments
    assert fcc is not f