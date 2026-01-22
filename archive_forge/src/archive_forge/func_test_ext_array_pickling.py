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
def test_ext_array_pickling(pickle_module):
    for proto in range(0, pickle_module.HIGHEST_PROTOCOL + 1):
        ty = ParamExtType(3)
        storage = pa.array([b'foo', b'bar'], type=pa.binary(3))
        arr = pa.ExtensionArray.from_storage(ty, storage)
        ser = pickle_module.dumps(arr, protocol=proto)
        del ty, storage, arr
        arr = pickle_module.loads(ser)
        arr.validate()
        assert isinstance(arr, pa.ExtensionArray)
        assert arr.type == ParamExtType(3)
        assert arr.type.storage_type == pa.binary(3)
        assert arr.storage.type == pa.binary(3)
        assert arr.storage.to_pylist() == [b'foo', b'bar']