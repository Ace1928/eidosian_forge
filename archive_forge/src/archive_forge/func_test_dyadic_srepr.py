from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.vector import (CoordSys3D, Vector, Dyadic,
def test_dyadic_srepr():
    from sympy.printing.repr import srepr
    N = CoordSys3D('N')
    dy = N.i | N.j
    res = "BaseDyadic(CoordSys3D(Str('N'), Tuple(ImmutableDenseMatrix([[Integer(1), Integer(0), Integer(0)], [Integer(0), Integer(1), Integer(0)], [Integer(0), Integer(0), Integer(1)]]), VectorZero())).i, CoordSys3D(Str('N'), Tuple(ImmutableDenseMatrix([[Integer(1), Integer(0), Integer(0)], [Integer(0), Integer(1), Integer(0)], [Integer(0), Integer(0), Integer(1)]]), VectorZero())).j)"
    assert srepr(dy) == res