import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_arithmetic_fused(self):
    vdata_a, vdata_b, vdata_c = [self.load(self._data())] * 3
    vdata_cx2 = self.add(vdata_c, vdata_c)
    data_fma = self.load([a * b + c for a, b, c in zip(vdata_a, vdata_b, vdata_c)])
    fma = self.muladd(vdata_a, vdata_b, vdata_c)
    assert fma == data_fma
    fms = self.mulsub(vdata_a, vdata_b, vdata_c)
    data_fms = self.sub(data_fma, vdata_cx2)
    assert fms == data_fms
    nfma = self.nmuladd(vdata_a, vdata_b, vdata_c)
    data_nfma = self.sub(vdata_cx2, data_fma)
    assert nfma == data_nfma
    nfms = self.nmulsub(vdata_a, vdata_b, vdata_c)
    data_nfms = self.mul(data_fma, self.setall(-1))
    assert nfms == data_nfms
    fmas = list(self.muladdsub(vdata_a, vdata_b, vdata_c))
    assert fmas[0::2] == list(data_fms)[0::2]
    assert fmas[1::2] == list(data_fma)[1::2]