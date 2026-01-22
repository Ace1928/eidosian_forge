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
def test_max_connections(self):
    with HTTPConnectionPool(host='localhost', maxsize=1, block=True) as pool:
        pool._get_conn(timeout=SHORT_TIMEOUT)
        with pytest.raises(EmptyPoolError):
            pool._get_conn(timeout=SHORT_TIMEOUT)
        with pytest.raises(EmptyPoolError):
            pool.request('GET', '/', pool_timeout=SHORT_TIMEOUT)
        assert pool.num_connections == 1