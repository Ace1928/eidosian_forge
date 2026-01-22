import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
@pytest.mark.parametrize('other_DT', simple_dtypes)
@pytest.mark.parametrize('string_char', ['S', 'U'])
def test_string_cancast(self, other_DT, string_char):
    fact = 1 if string_char == 'S' else 4
    string_DT = type(np.dtype(string_char))
    cast = get_castingimpl(other_DT, string_DT)
    other_dt = other_DT()
    expected_length = get_expected_stringlength(other_dt)
    string_dt = np.dtype(f'{string_char}{expected_length}')
    safety, (res_other_dt, res_dt), view_off = cast._resolve_descriptors((other_dt, None))
    assert res_dt.itemsize == expected_length * fact
    assert safety == Casting.safe
    assert view_off is None
    assert isinstance(res_dt, string_DT)
    for change_length in [-1, 0, 1]:
        if change_length >= 0:
            expected_safety = Casting.safe
        else:
            expected_safety = Casting.same_kind
        to_dt = self.string_with_modified_length(string_dt, change_length)
        safety, (_, res_dt), view_off = cast._resolve_descriptors((other_dt, to_dt))
        assert res_dt is to_dt
        assert safety == expected_safety
        assert view_off is None
    cast = get_castingimpl(string_DT, other_DT)
    safety, _, view_off = cast._resolve_descriptors((string_dt, other_dt))
    assert safety == Casting.unsafe
    assert view_off is None
    cast = get_castingimpl(string_DT, other_DT)
    safety, (_, res_dt), view_off = cast._resolve_descriptors((string_dt, None))
    assert safety == Casting.unsafe
    assert view_off is None
    assert other_dt is res_dt