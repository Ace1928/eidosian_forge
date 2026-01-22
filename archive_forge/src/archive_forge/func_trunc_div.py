import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def trunc_div(a, d):
    """
            Divide towards zero works with large integers > 2^53,
            and wrap around overflow similar to what C does.
            """
    if d == -1 and a == int_min:
        return a
    sign_a, sign_d = (a < 0, d < 0)
    if a == 0 or sign_a == sign_d:
        return a // d
    return (a + sign_d - sign_a) // d + 1