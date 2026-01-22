import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
@pytest.mark.parametrize('func, intrin', [(operator.lt, 'cmplt'), (operator.le, 'cmple'), (operator.gt, 'cmpgt'), (operator.ge, 'cmpge'), (operator.eq, 'cmpeq')])
def test_operators_comparison(self, func, intrin):
    if self._is_fp():
        data_a = self._data()
    else:
        data_a = self._data(self._int_max() - self.nlanes)
    data_b = self._data(self._int_min(), reverse=True)
    vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
    intrin = getattr(self, intrin)
    mask_true = self._true_mask()

    def to_bool(vector):
        return [lane == mask_true for lane in vector]
    data_cmp = [func(a, b) for a, b in zip(data_a, data_b)]
    cmp = to_bool(intrin(vdata_a, vdata_b))
    assert cmp == data_cmp