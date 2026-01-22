import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
def test_merge_pool_kwargs(self):
    """Assert _merge_pool_kwargs works in the happy case"""
    p = PoolManager(strict=True)
    merged = p._merge_pool_kwargs({'new_key': 'value'})
    assert {'strict': True, 'new_key': 'value'} == merged