import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
@resolvesLocalhostFQDN
def test_same_url(self):
    conn1 = connection_from_url('http://localhost:8081/foo')
    conn2 = connection_from_url('http://localhost:8081/bar')
    assert conn1 != conn2
    p = PoolManager(1)
    conn1 = p.connection_from_url('http://localhost:8081/foo')
    conn2 = p.connection_from_url('http://localhost:8081/bar')
    assert conn1 == conn2
    p = PoolManager(2)
    conn1 = p.connection_from_url('http://localhost.:8081/foo')
    conn2 = p.connection_from_url('http://localhost:8081/bar')
    assert conn1 != conn2