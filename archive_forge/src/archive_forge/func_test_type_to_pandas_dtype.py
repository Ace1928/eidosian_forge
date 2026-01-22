from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_type_to_pandas_dtype():
    M8 = np.dtype('datetime64[ms]')
    if Version(pd.__version__) < Version('2.0.0'):
        M8 = np.dtype('datetime64[ns]')
    cases = [(pa.null(), np.object_), (pa.bool_(), np.bool_), (pa.int8(), np.int8), (pa.int16(), np.int16), (pa.int32(), np.int32), (pa.int64(), np.int64), (pa.uint8(), np.uint8), (pa.uint16(), np.uint16), (pa.uint32(), np.uint32), (pa.uint64(), np.uint64), (pa.float16(), np.float16), (pa.float32(), np.float32), (pa.float64(), np.float64), (pa.date32(), M8), (pa.date64(), M8), (pa.timestamp('ms'), M8), (pa.binary(), np.object_), (pa.binary(12), np.object_), (pa.string(), np.object_), (pa.list_(pa.int8()), np.object_), (pa.map_(pa.int64(), pa.float64()), np.object_)]
    for arrow_type, numpy_type in cases:
        assert arrow_type.to_pandas_dtype() == numpy_type