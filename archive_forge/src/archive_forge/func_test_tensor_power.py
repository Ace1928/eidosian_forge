from sympy.physics.quantum.hilbert import (
from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.printing.repr import srepr
from sympy.printing.str import sstr
from sympy.sets.sets import Interval
def test_tensor_power():
    n = Symbol('n')
    hs1 = ComplexSpace(2)
    hs2 = ComplexSpace(n)
    h = hs1 ** 2
    assert isinstance(h, TensorPowerHilbertSpace)
    assert h.base == hs1
    assert h.exp == 2
    assert h.dimension == 4
    h = hs2 ** 3
    assert isinstance(h, TensorPowerHilbertSpace)
    assert h.base == hs2
    assert h.exp == 3
    assert h.dimension == n ** 3