from mpmath import *
from mpmath.libmp import *
import random
def test_pow_integer_direction():
    """
    Test that inexact integer powers are rounded in the right
    direction.
    """
    random.seed(1234)
    for prec in [10, 53, 200]:
        for i in range(50):
            a = random.randint(1 << prec - 1, 1 << prec)
            b = random.randint(2, 100)
            ab = a ** b
            assert to_int(mpf_pow(from_int(a), from_int(b), prec, round_down)) < ab
            assert to_int(mpf_pow(from_int(a), from_int(b), prec, round_up)) > ab