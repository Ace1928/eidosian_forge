import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
@pytest.mark.parametrize('start', [-100, -10000, 0, 100, 10000])
def test_reduce_max_min(self, start):
    """
        Test intrinsics:
            npyv_reduce_max_##sfx
            npyv_reduce_min_##sfx
        """
    vdata_a = self.load(self._data(start))
    assert self.reduce_max(vdata_a) == max(vdata_a)
    assert self.reduce_min(vdata_a) == min(vdata_a)