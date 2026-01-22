import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
def test_default_pool_key_funcs_copy(self):
    """Assert each PoolManager gets a copy of ``pool_keys_by_scheme``."""
    p = PoolManager()
    assert p.key_fn_by_scheme == p.key_fn_by_scheme
    assert p.key_fn_by_scheme is not key_fn_by_scheme