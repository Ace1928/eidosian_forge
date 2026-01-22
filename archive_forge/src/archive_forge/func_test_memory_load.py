import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_memory_load(self):
    data = self._data()
    load_data = self.load(data)
    assert load_data == data
    loada_data = self.loada(data)
    assert loada_data == data
    loads_data = self.loads(data)
    assert loads_data == data
    loadl = self.loadl(data)
    loadl_half = list(loadl)[:self.nlanes // 2]
    data_half = data[:self.nlanes // 2]
    assert loadl_half == data_half
    assert loadl != data