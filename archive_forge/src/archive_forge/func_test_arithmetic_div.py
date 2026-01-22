import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_arithmetic_div(self):
    if not self._is_fp():
        return
    data_a, data_b = (self._data(), self._data(reverse=True))
    vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
    data_div = self.load([a / b for a, b in zip(data_a, data_b)])
    div = self.div(vdata_a, vdata_b)
    assert div == data_div