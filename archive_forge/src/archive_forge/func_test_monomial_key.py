from sympy.polys.orderings import (
from sympy.abc import x, y, z, t
from sympy.core import S
from sympy.testing.pytest import raises
def test_monomial_key():
    assert monomial_key() == lex
    assert monomial_key('lex') == lex
    assert monomial_key('grlex') == grlex
    assert monomial_key('grevlex') == grevlex
    raises(ValueError, lambda: monomial_key('foo'))
    raises(ValueError, lambda: monomial_key(1))
    M = [x, x ** 2 * z ** 2, x * y, x ** 2, S.One, y ** 2, x ** 3, y, z, x * y ** 2 * z, x ** 2 * y ** 2]
    assert sorted(M, key=monomial_key('lex', [z, y, x])) == [S.One, x, x ** 2, x ** 3, y, x * y, y ** 2, x ** 2 * y ** 2, z, x * y ** 2 * z, x ** 2 * z ** 2]
    assert sorted(M, key=monomial_key('grlex', [z, y, x])) == [S.One, x, y, z, x ** 2, x * y, y ** 2, x ** 3, x ** 2 * y ** 2, x * y ** 2 * z, x ** 2 * z ** 2]
    assert sorted(M, key=monomial_key('grevlex', [z, y, x])) == [S.One, x, y, z, x ** 2, x * y, y ** 2, x ** 3, x ** 2 * y ** 2, x ** 2 * z ** 2, x * y ** 2 * z]