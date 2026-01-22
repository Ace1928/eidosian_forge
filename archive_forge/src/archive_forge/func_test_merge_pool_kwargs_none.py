import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
def test_merge_pool_kwargs_none(self):
    """Assert false-y values to _merge_pool_kwargs result in defaults"""
    p = PoolManager(strict=True)
    merged = p._merge_pool_kwargs({})
    assert p.connection_pool_kw == merged
    merged = p._merge_pool_kwargs(None)
    assert p.connection_pool_kw == merged