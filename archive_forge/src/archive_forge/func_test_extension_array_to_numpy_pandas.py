import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
@pytest.mark.parametrize('data,ty', (([1, 2, 3], IntegerType), (['cat', 'dog', 'horse'], LabelType)))
@pytest.mark.parametrize('into', ['to_numpy', pytest.param('to_pandas', marks=pytest.mark.pandas)])
def test_extension_array_to_numpy_pandas(data, ty, into):
    storage = pa.array(data)
    ext_arr = pa.ExtensionArray.from_storage(ty(), storage)
    offsets = pa.array([0, 1, 2, 3])
    list_arr = pa.ListArray.from_arrays(offsets, ext_arr)
    result = getattr(list_arr, into)(zero_copy_only=False)
    list_arr_storage_type = list_arr.cast(pa.list_(ext_arr.type.storage_type))
    expected = getattr(list_arr_storage_type, into)(zero_copy_only=False)
    if into == 'to_pandas':
        assert result.equals(expected)
    else:
        assert np.array_equal(result, expected)