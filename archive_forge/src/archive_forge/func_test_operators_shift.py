import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_operators_shift(self):
    if self.sfx in ('u8', 's8'):
        return
    data_a = self._data(self._int_max() - self.nlanes)
    data_b = self._data(self._int_min(), reverse=True)
    vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
    for count in range(self._scalar_size()):
        data_shl_a = self.load([a << count for a in data_a])
        shl = self.shl(vdata_a, count)
        assert shl == data_shl_a
        data_shr_a = self.load([a >> count for a in data_a])
        shr = self.shr(vdata_a, count)
        assert shr == data_shr_a
    for count in range(1, self._scalar_size()):
        data_shl_a = self.load([a << count for a in data_a])
        shli = self.shli(vdata_a, count)
        assert shli == data_shl_a
        data_shr_a = self.load([a >> count for a in data_a])
        shri = self.shri(vdata_a, count)
        assert shri == data_shr_a