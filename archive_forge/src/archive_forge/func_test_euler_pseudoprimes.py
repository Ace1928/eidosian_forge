from sympy.ntheory.generate import Sieve, sieve
from sympy.ntheory.primetest import (mr, is_lucas_prp, is_square,
from sympy.testing.pytest import slow
from sympy.core.numbers import I
def test_euler_pseudoprimes():
    assert is_euler_pseudoprime(9, 1) == True
    assert is_euler_pseudoprime(341, 2) == False
    assert is_euler_pseudoprime(121, 3) == True
    assert is_euler_pseudoprime(341, 4) == True
    assert is_euler_pseudoprime(217, 5) == False
    assert is_euler_pseudoprime(185, 6) == False
    assert is_euler_pseudoprime(55, 111) == True
    assert is_euler_pseudoprime(115, 114) == True
    assert is_euler_pseudoprime(49, 117) == True
    assert is_euler_pseudoprime(85, 84) == True
    assert is_euler_pseudoprime(87, 88) == True
    assert is_euler_pseudoprime(49, 128) == True
    assert is_euler_pseudoprime(39, 77) == True
    assert is_euler_pseudoprime(9881, 30) == True
    assert is_euler_pseudoprime(8841, 29) == False
    assert is_euler_pseudoprime(8421, 29) == False
    assert is_euler_pseudoprime(9997, 19) == True