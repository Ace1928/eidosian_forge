import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
@pytest.mark.filterwarnings('ignore::numpy.ComplexWarning')
@pytest.mark.parametrize('from_dt', simple_dtype_instances())
def test_simple_direct_casts(self, from_dt):
    """
        This test checks numeric direct casts for dtypes supported also by the
        struct module (plus complex).  It tries to be test a wide range of
        inputs, but skips over possibly undefined behaviour (e.g. int rollover).
        Longdouble and CLongdouble are tested, but only using double precision.

        If this test creates issues, it should possibly just be simplified
        or even removed (checking whether unaligned/non-contiguous casts give
        the same results is useful, though).
        """
    for to_dt in simple_dtype_instances():
        to_dt = to_dt.values[0]
        cast = get_castingimpl(type(from_dt), type(to_dt))
        casting, (from_res, to_res), view_off = cast._resolve_descriptors((from_dt, to_dt))
        if from_res is not from_dt or to_res is not to_dt:
            return
        safe = casting <= Casting.safe
        del from_res, to_res, casting
        arr1, arr2, values = self.get_data(from_dt, to_dt)
        cast._simple_strided_call((arr1, arr2))
        assert arr2.tolist() == values
        arr1_o, arr2_o = self.get_data_variation(arr1, arr2, True, False)
        cast._simple_strided_call((arr1_o, arr2_o))
        assert_array_equal(arr2_o, arr2)
        assert arr2_o.tobytes() == arr2.tobytes()
        if from_dt.alignment == 1 and to_dt.alignment == 1 or not cast._supports_unaligned:
            return
        arr1_o, arr2_o = self.get_data_variation(arr1, arr2, False, True)
        cast._simple_strided_call((arr1_o, arr2_o))
        assert_array_equal(arr2_o, arr2)
        assert arr2_o.tobytes() == arr2.tobytes()
        arr1_o, arr2_o = self.get_data_variation(arr1, arr2, False, False)
        cast._simple_strided_call((arr1_o, arr2_o))
        assert_array_equal(arr2_o, arr2)
        assert arr2_o.tobytes() == arr2.tobytes()
        del arr1_o, arr2_o, cast