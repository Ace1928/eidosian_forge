from __future__ import annotations
from sympy.ntheory import qs
from sympy.ntheory.qs import SievePolynomial, _generate_factor_base, \
from sympy.testing.pytest import slow
@slow
def test_qs_1():
    assert qs(10009202107, 100, 10000) == {100043, 100049}
    assert qs(211107295182713951054568361, 1000, 10000) == {13791315212531, 15307263442931}
    assert qs(980835832582657 * 990377764891511, 3000, 50000) == {980835832582657, 990377764891511}
    assert qs(18640889198609 * 20991129234731, 1000, 50000) == {18640889198609, 20991129234731}