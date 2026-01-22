import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_scalar_udf_context(unary_func_fixture):
    proxy_pool = pa.proxy_memory_pool(pa.default_memory_pool())
    _, func_name = unary_func_fixture
    res = pc.call_function(func_name, [pa.array([1] * 1000, type=pa.int64())], memory_pool=proxy_pool)
    assert res == pa.array([2] * 1000, type=pa.int64())
    assert proxy_pool.bytes_allocated() == 1000 * 8
    res = None
    assert proxy_pool.bytes_allocated() == 0