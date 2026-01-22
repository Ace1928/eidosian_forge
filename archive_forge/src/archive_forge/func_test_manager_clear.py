import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
def test_manager_clear(self):
    p = PoolManager(5)
    conn_pool = p.connection_from_url('http://google.com')
    assert len(p.pools) == 1
    conn = conn_pool._get_conn()
    p.clear()
    assert len(p.pools) == 0
    with pytest.raises(ClosedPoolError):
        conn_pool._get_conn()
    conn_pool._put_conn(conn)
    with pytest.raises(ClosedPoolError):
        conn_pool._get_conn()
    assert len(p.pools) == 0