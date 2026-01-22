from sympy.combinatorics.group_numbers import (is_nilpotent_number,
from sympy.testing.pytest import raises
from sympy import randprime
def test_is_abelian_number():
    assert is_abelian_number(4) == True
    assert is_abelian_number(randprime(1, 2000) ** 2) == True
    assert is_abelian_number(randprime(1000, 100000)) == True
    assert is_abelian_number(60) == False
    assert is_abelian_number(24) == False
    raises(ValueError, lambda: is_abelian_number(-5))