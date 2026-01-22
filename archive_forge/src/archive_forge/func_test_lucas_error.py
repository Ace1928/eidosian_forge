from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_lucas_error():
    raises(NotImplementedError, lambda: lucas(-1))