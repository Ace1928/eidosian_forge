import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
def make_extension_batch():
    schema = make_extension_schema()
    ext_col = schema[0].type.wrap_array(pa.array([b'foo', b'bar'], type=pa.binary(3)))
    return pa.record_batch([ext_col], schema)