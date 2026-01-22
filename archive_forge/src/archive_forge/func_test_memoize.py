from __future__ import annotations
from sympy.core.singleton import S
from sympy.core.basic import Basic
from sympy.strategies.core import (
from io import StringIO
def test_memoize():
    rl = memoize(posdec)
    assert rl(5) == posdec(5)
    assert rl(5) == posdec(5)
    assert rl(-2) == posdec(-2)