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
def test_release_conn_param_is_respected_after_http_error_retry(self):
    """For successful ```urlopen(release_conn=False)```,
        the connection isn't released, even after a retry.

        This is a regression test for issue #651 [1], where the connection
        would be released if the initial request failed, even if a retry
        succeeded.

        [1] <https://github.com/urllib3/urllib3/issues/651>
        """

    class _raise_once_make_request_function(object):
        """Callable that can mimic `_make_request()`.

            Raises the given exception on its first call, but returns a
            successful response on subsequent calls.
            """

        def __init__(self, ex):
            super(_raise_once_make_request_function, self).__init__()
            self._ex = ex

        def __call__(self, *args, **kwargs):
            if self._ex:
                ex, self._ex = (self._ex, None)
                raise ex()
            response = httplib.HTTPResponse(MockSock)
            response.fp = MockChunkedEncodingResponse([b'f', b'o', b'o'])
            response.headers = response.msg = HTTPHeaderDict()
            return response

    def _test(exception):
        with HTTPConnectionPool(host='localhost', maxsize=1, block=True) as pool:
            pool._make_request = _raise_once_make_request_function(exception)
            response = pool.urlopen('GET', '/', retries=1, release_conn=False, preload_content=False, chunked=True)
            assert pool.pool.qsize() == 0
            assert pool.num_connections == 2
            assert response.connection is not None
            response.release_conn()
            assert pool.pool.qsize() == 1
            assert response.connection is None
    _test(TimeoutError)
    _test(HTTPException)
    _test(SocketError)
    _test(ProtocolError)