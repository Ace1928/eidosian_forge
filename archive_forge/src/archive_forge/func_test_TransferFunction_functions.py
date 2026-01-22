from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import eye
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import CRootOf
from sympy.simplify.simplify import simplify
from sympy.core.containers import Tuple
from sympy.matrices import ImmutableMatrix, Matrix
from sympy.physics.control import (TransferFunction, Series, Parallel,
from sympy.testing.pytest import raises
def test_TransferFunction_functions():
    expr_1 = Mul(0, Pow(s, -1, evaluate=False), evaluate=False)
    expr_2 = s / 0
    expr_3 = (p * s ** 2 + 5 * s) / (s + 1) ** 3
    expr_4 = 6
    expr_5 = (2 + 3 * s) * (5 + 2 * s) / ((9 + 3 * s) * (5 + 2 * s ** 2))
    expr_6 = (9 * s ** 4 + 4 * s ** 2 + 8) / ((s + 1) * (s + 9))
    tf = TransferFunction(s + 1, s ** 2 + 2, s)
    delay = exp(-s / tau)
    expr_7 = delay * tf.to_expr()
    H1 = TransferFunction.from_rational_expression(expr_7, s)
    H2 = TransferFunction(s + 1, (s ** 2 + 2) * exp(s / tau), s)
    expr_8 = Add(2, 3 * s / (s ** 2 + 1), evaluate=False)
    assert TransferFunction.from_rational_expression(expr_1) == TransferFunction(0, s, s)
    raises(ZeroDivisionError, lambda: TransferFunction.from_rational_expression(expr_2))
    raises(ValueError, lambda: TransferFunction.from_rational_expression(expr_3))
    assert TransferFunction.from_rational_expression(expr_3, s) == TransferFunction(p * s ** 2 + 5 * s, (s + 1) ** 3, s)
    assert TransferFunction.from_rational_expression(expr_3, p) == TransferFunction(p * s ** 2 + 5 * s, (s + 1) ** 3, p)
    raises(ValueError, lambda: TransferFunction.from_rational_expression(expr_4))
    assert TransferFunction.from_rational_expression(expr_4, s) == TransferFunction(6, 1, s)
    assert TransferFunction.from_rational_expression(expr_5, s) == TransferFunction((2 + 3 * s) * (5 + 2 * s), (9 + 3 * s) * (5 + 2 * s ** 2), s)
    assert TransferFunction.from_rational_expression(expr_6, s) == TransferFunction(9 * s ** 4 + 4 * s ** 2 + 8, (s + 1) * (s + 9), s)
    assert H1 == H2
    assert TransferFunction.from_rational_expression(expr_8, s) == TransferFunction(2 * s ** 2 + 3 * s + 2, s ** 2 + 1, s)
    tf0 = TransferFunction(s ** 5 + s ** 3 + s, s - s ** 2, s)
    a = TransferFunction(-(s ** 4 + s ** 2 + 1), s - 1, s)
    assert tf0.simplify() == simplify(tf0) == a
    tf1 = TransferFunction((p + 3) * (p - 1), (p - 1) * (p + 5), p)
    b = TransferFunction(p + 3, p + 5, p)
    assert tf1.simplify() == simplify(tf1) == b
    G1 = TransferFunction((1 - s) ** 2, (s ** 2 + 1) ** 2, s)
    G2 = TransferFunction(1, -3, p)
    c = (a2 * s ** p + a1 * s ** s + a0 * p ** p) * (p ** s + s ** p)
    d = (b0 * s ** s + b1 * p ** s) * (b2 * s * p + p ** p)
    e = a0 * p ** p * p ** s + a0 * p ** p * s ** p + a1 * p ** s * s ** s + a1 * s ** p * s ** s + a2 * p ** s * s ** p + a2 * s ** (2 * p)
    f = b0 * b2 * p * s * s ** s + b0 * p ** p * s ** s + b1 * b2 * p * p ** s * s + b1 * p ** p * p ** s
    g = a1 * a2 * s * s ** p + a1 * p * s + a2 * b1 * p * s * s ** p + b1 * p ** 2 * s
    G3 = TransferFunction(c, d, s)
    G4 = TransferFunction(a0 * s ** s - b0 * p ** p, (a1 * s + b1 * s * p) * (a2 * s ** p + p), p)
    assert G1.expand() == TransferFunction(s ** 2 - 2 * s + 1, s ** 4 + 2 * s ** 2 + 1, s)
    assert tf1.expand() == TransferFunction(p ** 2 + 2 * p - 3, p ** 2 + 4 * p - 5, p)
    assert G2.expand() == G2
    assert G3.expand() == TransferFunction(e, f, s)
    assert G4.expand() == TransferFunction(a0 * s ** s - b0 * p ** p, g, p)
    p1 = a1 * s + a0
    p2 = b2 * s ** 2 + b1 * s + b0
    SP1 = TransferFunction(p1, p2, s)
    expect1 = TransferFunction(2.0 * s + 1.0, 5.0 * s ** 2 + 4.0 * s + 3.0, s)
    expect1_ = TransferFunction(2 * s + 1, 5 * s ** 2 + 4 * s + 3, s)
    assert SP1.subs({a0: 1, a1: 2, b0: 3, b1: 4, b2: 5}) == expect1_
    assert SP1.subs({a0: 1, a1: 2, b0: 3, b1: 4, b2: 5}).evalf() == expect1
    assert expect1_.evalf() == expect1
    c1, d0, d1, d2 = symbols('c1, d0:3')
    p3, p4 = (c1 * p, d2 * p ** 3 + d1 * p ** 2 - d0)
    SP2 = TransferFunction(p3, p4, p)
    expect2 = TransferFunction(2.0 * p, 5.0 * p ** 3 + 2.0 * p ** 2 - 3.0, p)
    expect2_ = TransferFunction(2 * p, 5 * p ** 3 + 2 * p ** 2 - 3, p)
    assert SP2.subs({c1: 2, d0: 3, d1: 2, d2: 5}) == expect2_
    assert SP2.subs({c1: 2, d0: 3, d1: 2, d2: 5}).evalf() == expect2
    assert expect2_.evalf() == expect2
    SP3 = TransferFunction(a0 * p ** 3 + a1 * s ** 2 - b0 * s + b1, a1 * s + p, s)
    expect3 = TransferFunction(2.0 * p ** 3 + 4.0 * s ** 2 - s + 5.0, p + 4.0 * s, s)
    expect3_ = TransferFunction(2 * p ** 3 + 4 * s ** 2 - s + 5, p + 4 * s, s)
    assert SP3.subs({a0: 2, a1: 4, b0: 1, b1: 5}) == expect3_
    assert SP3.subs({a0: 2, a1: 4, b0: 1, b1: 5}).evalf() == expect3
    assert expect3_.evalf() == expect3
    SP4 = TransferFunction(s - a1 * p ** 3, a0 * s + p, p)
    expect4 = TransferFunction(7.0 * p ** 3 + s, p - s, p)
    expect4_ = TransferFunction(7 * p ** 3 + s, p - s, p)
    assert SP4.subs({a0: -1, a1: -7}) == expect4_
    assert SP4.subs({a0: -1, a1: -7}).evalf() == expect4
    assert expect4_.evalf() == expect4
    assert tf0.dc_gain() == 1
    assert tf1.dc_gain() == Rational(3, 5)
    assert SP2.dc_gain() == 0
    assert expect4.dc_gain() == -1
    assert expect2_.dc_gain() == 0
    assert TransferFunction(1, s, s).dc_gain() == oo
    tf_ = TransferFunction(x ** 3 - k, k, x)
    _tf = TransferFunction(k, x ** 4 - k, x)
    TF_ = TransferFunction(x ** 2, x ** 10 + x + x ** 2, x)
    _TF = TransferFunction(x ** 10 + x + x ** 2, x ** 2, x)
    assert G1.poles() == [I, I, -I, -I]
    assert G2.poles() == []
    assert tf1.poles() == [-5, 1]
    assert expect4_.poles() == [s]
    assert SP4.poles() == [-a0 * s]
    assert expect3.poles() == [-0.25 * p]
    assert str(expect2.poles()) == str([0.729001428685125, -0.564500714342563 - 0.710198984796332 * I, -0.564500714342563 + 0.710198984796332 * I])
    assert str(expect1.poles()) == str([-0.4 - 0.66332495807108 * I, -0.4 + 0.66332495807108 * I])
    assert _tf.poles() == [k ** Rational(1, 4), -k ** Rational(1, 4), I * k ** Rational(1, 4), -I * k ** Rational(1, 4)]
    assert TF_.poles() == [CRootOf(x ** 9 + x + 1, 0), 0, CRootOf(x ** 9 + x + 1, 1), CRootOf(x ** 9 + x + 1, 2), CRootOf(x ** 9 + x + 1, 3), CRootOf(x ** 9 + x + 1, 4), CRootOf(x ** 9 + x + 1, 5), CRootOf(x ** 9 + x + 1, 6), CRootOf(x ** 9 + x + 1, 7), CRootOf(x ** 9 + x + 1, 8)]
    raises(NotImplementedError, lambda: TransferFunction(x ** 2, a0 * x ** 10 + x + x ** 2, x).poles())
    q, r = symbols('q, r', negative=True)
    t = symbols('t', positive=True)
    TF_ = TransferFunction(s ** 2 + a0 - a1 * p, q * s - r, s)
    stable_tf = TransferFunction(s ** 2 + a0 - a1 * p, q * s - 1, s)
    stable_tf_ = TransferFunction(s ** 2 + a0 - a1 * p, q * s - t, s)
    assert G1.is_stable() is False
    assert G2.is_stable() is True
    assert tf1.is_stable() is False
    assert expect2.is_stable() is False
    assert expect1.is_stable() is True
    assert stable_tf.is_stable() is True
    assert stable_tf_.is_stable() is True
    assert TF_.is_stable() is False
    assert expect4_.is_stable() is None
    assert SP4.is_stable() is None
    assert G1.zeros() == [1, 1]
    assert G2.zeros() == []
    assert tf1.zeros() == [-3, 1]
    assert expect4_.zeros() == [7 ** Rational(2, 3) * (-s) ** Rational(1, 3) / 7, -7 ** Rational(2, 3) * (-s) ** Rational(1, 3) / 14 - sqrt(3) * 7 ** Rational(2, 3) * I * (-s) ** Rational(1, 3) / 14, -7 ** Rational(2, 3) * (-s) ** Rational(1, 3) / 14 + sqrt(3) * 7 ** Rational(2, 3) * I * (-s) ** Rational(1, 3) / 14]
    assert SP4.zeros() == [(s / a1) ** Rational(1, 3), -(s / a1) ** Rational(1, 3) / 2 - sqrt(3) * I * (s / a1) ** Rational(1, 3) / 2, -(s / a1) ** Rational(1, 3) / 2 + sqrt(3) * I * (s / a1) ** Rational(1, 3) / 2]
    assert str(expect3.zeros()) == str([0.125 - 1.11102430216445 * sqrt(-0.405063291139241 * p ** 3 - 1.0), 1.11102430216445 * sqrt(-0.405063291139241 * p ** 3 - 1.0) + 0.125])
    assert tf_.zeros() == [k ** Rational(1, 3), -k ** Rational(1, 3) / 2 - sqrt(3) * I * k ** Rational(1, 3) / 2, -k ** Rational(1, 3) / 2 + sqrt(3) * I * k ** Rational(1, 3) / 2]
    assert _TF.zeros() == [CRootOf(x ** 9 + x + 1, 0), 0, CRootOf(x ** 9 + x + 1, 1), CRootOf(x ** 9 + x + 1, 2), CRootOf(x ** 9 + x + 1, 3), CRootOf(x ** 9 + x + 1, 4), CRootOf(x ** 9 + x + 1, 5), CRootOf(x ** 9 + x + 1, 6), CRootOf(x ** 9 + x + 1, 7), CRootOf(x ** 9 + x + 1, 8)]
    raises(NotImplementedError, lambda: TransferFunction(a0 * x ** 10 + x + x ** 2, x ** 2, x).zeros())
    tf2 = TransferFunction(s + 3, s ** 2 - s ** 3 + 9, s)
    tf3 = TransferFunction(-3 * p + 3, 1 - p, p)
    assert -tf2 == TransferFunction(-s - 3, s ** 2 - s ** 3 + 9, s)
    assert -tf3 == TransferFunction(3 * p - 3, 1 - p, p)
    tf4 = TransferFunction(p + 4, p - 3, p)
    tf5 = TransferFunction(s ** 2 + 1, 1 - s, s)
    expect2 = TransferFunction((s ** 2 + 1) ** 3, (1 - s) ** 3, s)
    expect1 = TransferFunction((p + 4) ** 2, (p - 3) ** 2, p)
    assert (tf4 * tf4).doit() == tf4 ** 2 == pow(tf4, 2) == expect1
    assert (tf5 * tf5 * tf5).doit() == tf5 ** 3 == pow(tf5, 3) == expect2
    assert tf5 ** 0 == pow(tf5, 0) == TransferFunction(1, 1, s)
    assert Series(tf4).doit() ** (-1) == tf4 ** (-1) == pow(tf4, -1) == TransferFunction(p - 3, p + 4, p)
    assert (tf5 * tf5).doit() ** (-1) == tf5 ** (-2) == pow(tf5, -2) == TransferFunction((1 - s) ** 2, (s ** 2 + 1) ** 2, s)
    raises(ValueError, lambda: tf4 ** (s ** 2 + s - 1))
    raises(ValueError, lambda: tf5 ** s)
    raises(ValueError, lambda: tf4 ** tf5)
    tf = TransferFunction(s - 1, s ** 2 - 2 * s + 1, s)
    tf6 = TransferFunction(s + p, p ** 2 - 5, s)
    assert factor(tf) == TransferFunction(s - 1, (s - 1) ** 2, s)
    assert tf.num.subs(s, 2) == tf.den.subs(s, 2) == 1
    assert tf.subs(s, 2) == TransferFunction(s - 1, s ** 2 - 2 * s + 1, s)
    assert tf6.subs(p, 3) == TransferFunction(s + 3, 4, s)
    assert tf3.xreplace({p: s}) == TransferFunction(-3 * s + 3, 1 - s, s)
    raises(TypeError, lambda: tf3.xreplace({p: exp(2)}))
    assert tf3.subs(p, exp(2)) == tf3
    tf7 = TransferFunction(a0 * s ** p + a1 * p ** s, a2 * p - s, s)
    assert tf7.xreplace({s: k}) == TransferFunction(a0 * k ** p + a1 * p ** k, a2 * p - k, k)
    assert tf7.subs(s, k) == TransferFunction(a0 * s ** p + a1 * p ** s, a2 * p - s, s)
    tf8 = TransferFunction(a0 * s ** 5 + 5 * s ** 2 + 3, s ** 6 - 3, s)
    tf9 = TransferFunction(5 + s, (5 + s) * (6 + s), s)
    tf10 = TransferFunction(0, 1, s)
    tf11 = TransferFunction(1, 1, s)
    assert tf8.to_expr() == Mul(a0 * s ** 5 + 5 * s ** 2 + 3, Pow(s ** 6 - 3, -1, evaluate=False), evaluate=False)
    assert tf9.to_expr() == Mul(s + 5, Pow((5 + s) * (6 + s), -1, evaluate=False), evaluate=False)
    assert tf10.to_expr() == Mul(S(0), Pow(1, -1, evaluate=False), evaluate=False)
    assert tf11.to_expr() == Pow(1, -1, evaluate=False)