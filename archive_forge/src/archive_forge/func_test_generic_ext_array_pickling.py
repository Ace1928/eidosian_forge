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
def test_generic_ext_array_pickling(registered_period_type, pickle_module):
    for proto in range(0, pickle_module.HIGHEST_PROTOCOL + 1):
        period_type, _ = registered_period_type
        storage = pa.array([1, 2, 3, 4], pa.int64())
        arr = pa.ExtensionArray.from_storage(period_type, storage)
        ser = pickle_module.dumps(arr, protocol=proto)
        del storage, arr
        arr = pickle_module.loads(ser)
        arr.validate()
        assert isinstance(arr, pa.ExtensionArray)
        assert arr.type == period_type
        assert arr.type.storage_type == pa.int64()
        assert arr.storage.type == pa.int64()
        assert arr.storage.to_pylist() == [1, 2, 3, 4]