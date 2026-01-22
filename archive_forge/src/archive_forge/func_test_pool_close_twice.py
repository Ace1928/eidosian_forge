from __future__ import absolute_import
import ssl
from socket import error as SocketError
from ssl import SSLError as BaseSSLError
from test import SHORT_TIMEOUT
import pytest
from mock import Mock
from dummyserver.server import DEFAULT_CA
from urllib3._collections import HTTPHeaderDict
from urllib3.connectionpool import (
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.packages.six.moves.http_client import HTTPException
from urllib3.packages.six.moves.queue import Empty
from urllib3.response import HTTPResponse
from urllib3.util.ssl_match_hostname import CertificateError
from urllib3.util.timeout import Timeout
from .test_response import MockChunkedEncodingResponse, MockSock
def test_pool_close_twice(self):
    pool = connection_from_url('http://google.com:80')
    conn1 = pool._get_conn()
    conn2 = pool._get_conn()
    pool._put_conn(conn1)
    pool._put_conn(conn2)
    pool.close()
    assert pool.pool is None
    try:
        pool.close()
    except AttributeError:
        pytest.fail('Pool of the ConnectionPool is None and has no attribute get.')