import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_mask_conditional(self):
    """
        Conditional addition and subtraction for all supported data types.
        Test intrinsics:
            npyv_ifadd_##SFX, npyv_ifsub_##SFX
        """
    vdata_a = self.load(self._data())
    vdata_b = self.load(self._data(reverse=True))
    true_mask = self.cmpeq(self.zero(), self.zero())
    false_mask = self.cmpneq(self.zero(), self.zero())
    data_sub = self.sub(vdata_b, vdata_a)
    ifsub = self.ifsub(true_mask, vdata_b, vdata_a, vdata_b)
    assert ifsub == data_sub
    ifsub = self.ifsub(false_mask, vdata_a, vdata_b, vdata_b)
    assert ifsub == vdata_b
    data_add = self.add(vdata_b, vdata_a)
    ifadd = self.ifadd(true_mask, vdata_b, vdata_a, vdata_b)
    assert ifadd == data_add
    ifadd = self.ifadd(false_mask, vdata_a, vdata_b, vdata_b)
    assert ifadd == vdata_b
    if not self._is_fp():
        return
    data_div = self.div(vdata_b, vdata_a)
    ifdiv = self.ifdiv(true_mask, vdata_b, vdata_a, vdata_b)
    assert ifdiv == data_div
    ifdivz = self.ifdivz(true_mask, vdata_b, vdata_a)
    assert ifdivz == data_div
    ifdiv = self.ifdiv(false_mask, vdata_a, vdata_b, vdata_b)
    assert ifdiv == vdata_b
    ifdivz = self.ifdivz(false_mask, vdata_a, vdata_b)
    assert ifdivz == self.zero()