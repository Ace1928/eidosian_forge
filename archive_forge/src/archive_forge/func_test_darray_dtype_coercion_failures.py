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
def test_darray_dtype_coercion_failures():
    dtypes = (np.uint8, np.int32, np.int64, np.float32, np.float64)
    encodings = ('ASCII', 'B64BIN', 'B64GZ')
    for data_dtype, darray_dtype, encoding in itertools.product(dtypes, dtypes, encodings):
        da = GiftiDataArray(np.arange(10, dtype=data_dtype), encoding=encoding, intent='NIFTI_INTENT_NODE_INDEX', datatype=darray_dtype)
        gii = GiftiImage(darrays=[da])
        gii_copy = GiftiImage.from_bytes(gii.to_bytes(mode='force'))
        da_copy = gii_copy.darrays[0]
        assert np.dtype(da_copy.data.dtype) == np.dtype(darray_dtype)
        assert_array_equal(da_copy.data, da.data)