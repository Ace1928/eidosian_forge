import ctypes
from functools import wraps
import pytest
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
def test_dlpack_not_supported():
    if Version(np.__version__) < Version('1.22.0'):
        pytest.skip('No dlpack support in numpy versions older than 1.22.0.')
    arr = pa.array([1, None, 3])
    with pytest.raises(TypeError, match='Can only use DLPack on arrays with no nulls.'):
        np.from_dlpack(arr)
    arr = pa.array([[0, 1], [3, 4]], type=pa.list_(pa.int32()))
    with pytest.raises(TypeError, match='DataType is not compatible with DLPack spec'):
        np.from_dlpack(arr)
    arr = pa.array([])
    with pytest.raises(TypeError, match='DataType is not compatible with DLPack spec'):
        np.from_dlpack(arr)
    arr = pa.array([True, False, True])
    with pytest.raises(TypeError, match='Bit-packed boolean data type not supported by DLPack.'):
        np.from_dlpack(arr)