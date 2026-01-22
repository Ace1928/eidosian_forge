import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
def make_extension_storage_schema():
    return pa.schema([('ext', ParamExtType(3).storage_type)], metadata={b'key1': b'value1'})