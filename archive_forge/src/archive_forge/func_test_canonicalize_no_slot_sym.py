from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, eye)
from sympy.tensor.indexed import Indexed
from sympy.combinatorics import Permutation
from sympy.core import S, Rational, Symbol, Basic, Add
from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.tensor.array import Array
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, \
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.matrices import diag
def test_canonicalize_no_slot_sym():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, d0, d1 = tensor_indices('a,b,d0,d1', Lorentz)
    A, B = tensor_heads('A,B', [Lorentz], TensorSymmetry.no_symmetry(1))
    t = A(-d0) * B(d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0)*B(-L_0)'
    t = A(a) * B(b)
    tc = t.canon_bp()
    assert tc == t
    t1 = B(b) * A(a)
    tc = t1.canon_bp()
    assert str(tc) == 'A(a)*B(b)'
    A = TensorHead('A', [Lorentz] * 2, TensorSymmetry.fully_symmetric(2))
    t = A(b, -d0) * A(d0, a)
    tc = t.canon_bp()
    assert str(tc) == 'A(a, L_0)*A(b, -L_0)'
    B, C = tensor_heads('B,C', [Lorentz], TensorSymmetry.no_symmetry(1))
    t = A(d1, -d0) * B(d0) * C(-d1)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-L_0)*C(-L_1)'
    A = TensorHead('A', [Lorentz] * 2, TensorSymmetry.no_symmetry(2))
    t = A(d1, -d0) * B(d0) * C(-d1)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-L_1)*C(-L_0)'
    B = TensorHead('B', [Lorentz] * 2, TensorSymmetry.no_symmetry(2))
    t = A(d1, -d0) * B(-d1, d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-L_0, -L_1)'
    t = A(-d0, d1) * B(-d1, d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-L_1, -L_0)'
    C = TensorHead('C', [Lorentz] * 2, TensorSymmetry.no_symmetry(2))
    t = A(d1, d0) * B(-a, -d0) * C(-d1, -b)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-a, -L_1)*C(-L_0, -b)'
    A = TensorHead('A', [Lorentz] * 2, TensorSymmetry.fully_symmetric(2))
    t = A(d1, d0) * B(-a, -d0) * C(-d1, -b)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-a, -L_0)*C(-L_1, -b)'
    C = TensorHead('C', [Lorentz] * 2, TensorSymmetry.fully_symmetric(2))
    t = A(d1, d0) * B(-a, -d0) * C(-d1, -b)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-a, -L_0)*C(-b, -L_1)'