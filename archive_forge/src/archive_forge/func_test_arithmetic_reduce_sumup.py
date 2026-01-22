import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_arithmetic_reduce_sumup(self):
    """
        Test extend reduce sum intrinsics:
            npyv_sumup_##sfx
        """
    if self.sfx not in ('u8', 'u16'):
        return
    rdata = (0, self.nlanes, self._int_min(), self._int_max() - self.nlanes)
    for r in rdata:
        data = self._data(r)
        vdata = self.load(data)
        data_sum = sum(data)
        vsum = self.sumup(vdata)
        assert vsum == data_sum