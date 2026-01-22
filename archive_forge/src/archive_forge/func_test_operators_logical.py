import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_operators_logical(self):
    if self._is_fp():
        data_a = self._data()
    else:
        data_a = self._data(self._int_max() - self.nlanes)
    data_b = self._data(self._int_min(), reverse=True)
    vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
    if self._is_fp():
        data_cast_a = self._to_unsigned(vdata_a)
        data_cast_b = self._to_unsigned(vdata_b)
        cast, cast_data = (self._to_unsigned, self._to_unsigned)
    else:
        data_cast_a, data_cast_b = (data_a, data_b)
        cast, cast_data = (lambda a: a, self.load)
    data_xor = cast_data([a ^ b for a, b in zip(data_cast_a, data_cast_b)])
    vxor = cast(self.xor(vdata_a, vdata_b))
    assert vxor == data_xor
    data_or = cast_data([a | b for a, b in zip(data_cast_a, data_cast_b)])
    vor = cast(getattr(self, 'or')(vdata_a, vdata_b))
    assert vor == data_or
    data_and = cast_data([a & b for a, b in zip(data_cast_a, data_cast_b)])
    vand = cast(getattr(self, 'and')(vdata_a, vdata_b))
    assert vand == data_and
    data_not = cast_data([~a for a in data_cast_a])
    vnot = cast(getattr(self, 'not')(vdata_a))
    assert vnot == data_not
    if self.sfx not in 'u8':
        return
    data_andc = [a & ~b for a, b in zip(data_cast_a, data_cast_b)]
    vandc = cast(getattr(self, 'andc')(vdata_a, vdata_b))
    assert vandc == data_andc