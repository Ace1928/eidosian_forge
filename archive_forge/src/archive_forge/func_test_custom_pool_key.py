import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
def test_custom_pool_key(self):
    """Assert it is possible to define a custom key function."""
    p = PoolManager(10)
    p.key_fn_by_scheme['http'] = lambda x: tuple(x['key'])
    pool1 = p.connection_from_url('http://example.com', pool_kwargs={'key': 'value'})
    pool2 = p.connection_from_url('http://example.com', pool_kwargs={'key': 'other'})
    pool3 = p.connection_from_url('http://example.com', pool_kwargs={'key': 'value', 'x': 'y'})
    assert 2 == len(p.pools)
    assert pool1 is pool3
    assert pool1 is not pool2