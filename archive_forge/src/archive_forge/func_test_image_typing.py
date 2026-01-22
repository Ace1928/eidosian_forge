import itertools
import sys
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nibabel.tmpdirs import InTemporaryDirectory
from ... import load
from ...fileholders import FileHolder
from ...nifti1 import data_type_codes
from ...testing import get_test_data
from .. import (
from .test_parse_gifti_fast import (
@pytest.mark.parametrize('label', data_type_codes.value_set('label'))
def test_image_typing(label):
    dtype = data_type_codes.dtype[label]
    if dtype == np.void:
        return
    arr = 127 * rng.random(20)
    try:
        cast = arr.astype(label)
    except TypeError:
        return
    darr = GiftiDataArray(cast, datatype=label)
    img = GiftiImage(darrays=[darr])
    force_rt = img.from_bytes(img.to_bytes(mode='force'))
    assert np.array_equal(cast, force_rt.darrays[0].data)
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating):
        compat_rt = img.from_bytes(img.to_bytes(mode='compat'))
        compat_darr = compat_rt.darrays[0].data
        assert np.allclose(cast, compat_darr)
        assert compat_darr.dtype in ('uint8', 'int32', 'float32')
    else:
        with pytest.raises(ValueError):
            img.to_bytes(mode='compat')
    if label in ('uint8', 'int32', 'float32'):
        strict_rt = img.from_bytes(img.to_bytes(mode='strict'))
        assert np.array_equal(cast, strict_rt.darrays[0].data)
    else:
        with pytest.raises(ValueError):
            img.to_bytes(mode='strict')