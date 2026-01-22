from __future__ import annotations
import pytest
import numpy as np
import pytest
from tlz import concat
import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.array.numpy_compat import AxisError
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('dtype', [np.uint8, np.int16, np.float32, bool])
@pytest.mark.parametrize('pad_widths', [2, (2,), (2, 3), ((2, 3),), ((3, 1), (0, 0), (2, 0))])
@pytest.mark.parametrize('mode', ['constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'minimum', pytest.param('reflect', marks=pytest.mark.skip(reason='Bug when pad_width is larger than dimension: https://github.com/dask/dask/issues/5303')), pytest.param('symmetric', marks=pytest.mark.skip(reason='Bug when pad_width is larger than dimension: https://github.com/dask/dask/issues/5303')), pytest.param('wrap', marks=pytest.mark.skip(reason='Bug when pad_width is larger than dimension: https://github.com/dask/dask/issues/5303')), pytest.param('median', marks=pytest.mark.skip(reason='Not implemented')), pytest.param('empty', marks=pytest.mark.skip(reason='Empty leads to undefined values, which may be different'))])
def test_pad_3d_data(dtype, pad_widths, mode):
    np_a = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(dtype)
    da_a = da.from_array(np_a, chunks='auto')
    np_r = np.pad(np_a, pad_widths, mode=mode)
    da_r = da.pad(da_a, pad_widths, mode=mode)
    assert_eq(np_r, da_r)