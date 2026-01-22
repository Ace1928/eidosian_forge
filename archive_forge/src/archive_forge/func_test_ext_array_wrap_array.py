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
def test_ext_array_wrap_array():
    ty = ParamExtType(3)
    storage = pa.array([b'foo', b'bar', None], type=pa.binary(3))
    arr = ty.wrap_array(storage)
    arr.validate(full=True)
    assert isinstance(arr, pa.ExtensionArray)
    assert arr.type == ty
    assert arr.storage == storage
    storage = pa.chunked_array([[b'abc', b'def'], [b'ghi']], type=pa.binary(3))
    arr = ty.wrap_array(storage)
    arr.validate(full=True)
    assert isinstance(arr, pa.ChunkedArray)
    assert arr.type == ty
    assert arr.chunk(0).storage == storage.chunk(0)
    assert arr.chunk(1).storage == storage.chunk(1)
    storage = pa.array([b'foo', b'bar', None])
    with pytest.raises(TypeError, match='Incompatible storage type'):
        ty.wrap_array(storage)
    with pytest.raises(TypeError, match='Expected array or chunked array'):
        ty.wrap_array(None)