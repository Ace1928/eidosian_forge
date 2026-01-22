import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def check_floatstatus(divbyzero=False, overflow=False, underflow=False, invalid=False, all=False):
    err = get_floatstatus()
    ret = (all or divbyzero) and err & 1 != 0
    ret |= (all or overflow) and err & 2 != 0
    ret |= (all or underflow) and err & 4 != 0
    ret |= (all or invalid) and err & 8 != 0
    return ret