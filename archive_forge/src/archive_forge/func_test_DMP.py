from __future__ import annotations
from typing import Any
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.assumptions.ask import Q
from sympy.core.function import (Function, WildFunction)
from sympy.core.numbers import (AlgebraicNumber, Float, Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import sin
from sympy.functions.special.delta_functions import Heaviside
from sympy.logic.boolalg import (false, true)
from sympy.matrices.dense import (Matrix, ones)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.combinatorics import Cycle, Permutation
from sympy.core.symbol import Str
from sympy.geometry import Point, Ellipse
from sympy.printing import srepr
from sympy.polys import ring, field, ZZ, QQ, lex, grlex, Poly
from sympy.polys.polyclasses import DMP
from sympy.polys.agca.extensions import FiniteExtension
def test_DMP():
    assert srepr(DMP([1, 2], ZZ)) == 'DMP([1, 2], ZZ)'
    assert srepr(ZZ.old_poly_ring(x)([1, 2])) == "DMP([1, 2], ZZ, ring=GlobalPolynomialRing(ZZ, Symbol('x')))"