import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
def make_batch():
    return pa.record_batch([[[1], [2, 42]]], make_schema())