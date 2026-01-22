import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_reorder_rev64(self):
    ssize = self._scalar_size()
    if ssize == 64:
        return
    data_rev64 = [y for x in range(0, self.nlanes, 64 // ssize) for y in reversed(range(x, x + 64 // ssize))]
    rev64 = self.rev64(self.load(range(self.nlanes)))
    assert rev64 == data_rev64