import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
@contextlib.contextmanager
def registered_extension_type(ext_type):
    pa.register_extension_type(ext_type)
    try:
        yield
    finally:
        pa.unregister_extension_type(ext_type.extension_name)