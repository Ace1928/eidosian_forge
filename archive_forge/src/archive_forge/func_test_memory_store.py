import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_memory_store(self):
    data = self._data()
    vdata = self.load(data)
    store = [0] * self.nlanes
    self.store(store, vdata)
    assert store == data
    store_a = [0] * self.nlanes
    self.storea(store_a, vdata)
    assert store_a == data
    store_s = [0] * self.nlanes
    self.stores(store_s, vdata)
    assert store_s == data
    store_l = [0] * self.nlanes
    self.storel(store_l, vdata)
    assert store_l[:self.nlanes // 2] == data[:self.nlanes // 2]
    assert store_l != vdata
    store_h = [0] * self.nlanes
    self.storeh(store_h, vdata)
    assert store_h[:self.nlanes // 2] == data[self.nlanes // 2:]
    assert store_h != vdata