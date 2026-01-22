import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
def test_merge_pool_kwargs_remove_key(self):
    """Assert keys can be removed with _merge_pool_kwargs"""
    p = PoolManager(strict=True)
    merged = p._merge_pool_kwargs({'strict': None})
    assert 'strict' not in merged