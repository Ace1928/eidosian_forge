import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
@pytest.mark.parametrize('dtype', simple_dtype_instances())
def test_object_and_simple_resolution(self, dtype):
    object_dtype = type(np.dtype(object))
    cast = get_castingimpl(object_dtype, type(dtype))
    safety, (_, res_dt), view_off = cast._resolve_descriptors((np.dtype('O'), dtype))
    assert safety == Casting.unsafe
    assert view_off is None
    assert res_dt is dtype
    safety, (_, res_dt), view_off = cast._resolve_descriptors((np.dtype('O'), None))
    assert safety == Casting.unsafe
    assert view_off is None
    assert res_dt == dtype.newbyteorder('=')