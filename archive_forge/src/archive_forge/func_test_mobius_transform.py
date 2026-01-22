from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Symbol, symbols, I, Rational
from sympy.discrete import (fft, ifft, ntt, intt, fwht, ifwht,
from sympy.testing.pytest import raises
def test_mobius_transform():
    assert all((tf(ls, subset=subset) == ls for ls in ([], [Rational(7, 4)]) for subset in (True, False) for tf in (mobius_transform, inverse_mobius_transform)))
    w, x, y, z = symbols('w x y z')
    assert mobius_transform([x, y]) == [x, x + y]
    assert inverse_mobius_transform([x, x + y]) == [x, y]
    assert mobius_transform([x, y], subset=False) == [x + y, y]
    assert inverse_mobius_transform([x + y, y], subset=False) == [x, y]
    assert mobius_transform([w, x, y, z]) == [w, w + x, w + y, w + x + y + z]
    assert inverse_mobius_transform([w, w + x, w + y, w + x + y + z]) == [w, x, y, z]
    assert mobius_transform([w, x, y, z], subset=False) == [w + x + y + z, x + z, y + z, z]
    assert inverse_mobius_transform([w + x + y + z, x + z, y + z, z], subset=False) == [w, x, y, z]
    ls = [Rational(2, 3), Rational(6, 7), Rational(5, 8), 9, Rational(5, 3) + 7 * I]
    mls = [Rational(2, 3), Rational(32, 21), Rational(31, 24), Rational(1873, 168), Rational(7, 3) + 7 * I, Rational(67, 21) + 7 * I, Rational(71, 24) + 7 * I, Rational(2153, 168) + 7 * I]
    assert mobius_transform(ls) == mls
    assert inverse_mobius_transform(mls) == ls + [S.Zero] * 3
    mls = [Rational(2153, 168) + 7 * I, Rational(69, 7), Rational(77, 8), 9, Rational(5, 3) + 7 * I, 0, 0, 0]
    assert mobius_transform(ls, subset=False) == mls
    assert inverse_mobius_transform(mls, subset=False) == ls + [S.Zero] * 3
    ls = ls[:-1]
    mls = [Rational(2, 3), Rational(32, 21), Rational(31, 24), Rational(1873, 168)]
    assert mobius_transform(ls) == mls
    assert inverse_mobius_transform(mls) == ls
    mls = [Rational(1873, 168), Rational(69, 7), Rational(77, 8), 9]
    assert mobius_transform(ls, subset=False) == mls
    assert inverse_mobius_transform(mls, subset=False) == ls
    raises(TypeError, lambda: mobius_transform(x, subset=True))
    raises(TypeError, lambda: inverse_mobius_transform(y, subset=False))