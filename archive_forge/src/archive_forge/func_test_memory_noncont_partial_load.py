import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
@pytest.mark.parametrize('intrin, elsizes, scale, fill', [('self.loadn_tillz, self.loadn_till', (32, 64), 1, [65535]), ('self.loadn2_tillz, self.loadn2_till', (32, 64), 2, [65535, 32767])])
def test_memory_noncont_partial_load(self, intrin, elsizes, scale, fill):
    if self._scalar_size() not in elsizes:
        return
    npyv_loadn_tillz, npyv_loadn_till = eval(intrin)
    lanes = list(range(1, self.nlanes + 1))
    lanes += [self.nlanes ** 2, self.nlanes ** 4]
    for stride in range(-64, 64):
        if stride < 0:
            data = self._data(stride, -stride * self.nlanes)
            data_stride = list(itertools.chain(*zip(*[data[-i::stride] for i in range(scale, 0, -1)])))
        elif stride == 0:
            data = self._data()
            data_stride = data[0:scale] * (self.nlanes // scale)
        else:
            data = self._data(count=stride * self.nlanes)
            data_stride = list(itertools.chain(*zip(*[data[i::stride] for i in range(scale)])))
        data_stride = list(self.load(data_stride))
        for n in lanes:
            nscale = n * scale
            llanes = self.nlanes - nscale
            data_stride_till = data_stride[:nscale] + fill * (llanes // scale)
            loadn_till = npyv_loadn_till(data, stride, n, *fill)
            assert loadn_till == data_stride_till
            data_stride_tillz = data_stride[:nscale] + [0] * llanes
            loadn_tillz = npyv_loadn_tillz(data, stride, n)
            assert loadn_tillz == data_stride_tillz