from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_factorial_error():
    raises(ArithmeticError, lambda: factorial(-1))