import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
@pytest.mark.parametrize('intrin, elsizes, scale', [('self.store_till', (32, 64), 1), ('self.store2_till', (32, 64), 2)])
def test_memory_partial_store(self, intrin, elsizes, scale):
    if self._scalar_size() not in elsizes:
        return
    npyv_store_till = eval(intrin)
    data = self._data()
    data_rev = self._data(reverse=True)
    vdata = self.load(data)
    lanes = list(range(1, self.nlanes + 1))
    lanes += [self.nlanes ** 2, self.nlanes ** 4]
    for n in lanes:
        data_till = data_rev.copy()
        data_till[:n * scale] = data[:n * scale]
        store_till = self._data(reverse=True)
        npyv_store_till(store_till, n, vdata)
        assert store_till == data_till