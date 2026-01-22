import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
def test_https_connection_from_context_case_insensitive(self):
    """Assert scheme case is ignored when getting the https key class."""
    p = PoolManager()
    context = {'scheme': 'https', 'host': 'example.com', 'port': '443'}
    other_context = {'scheme': 'HTTPS', 'host': 'EXAMPLE.COM', 'port': '443'}
    pool = p.connection_from_context(context)
    other_pool = p.connection_from_context(other_context)
    assert 1 == len(p.pools)
    assert pool is other_pool
    assert all((isinstance(key, PoolKey) for key in p.pools.keys()))