import contextlib
import re
import socket
import ssl
import zlib
from base64 import b64decode
from io import BufferedReader, BytesIO, TextIOWrapper
from test import onlyBrotlipy
import mock
import pytest
import six
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.response import HTTPResponse, brotli
from urllib3.util.response import is_fp_closed
from urllib3.util.retry import RequestHistory, Retry
def test_chunked_head_response(self):
    r = httplib.HTTPResponse(MockSock, method='HEAD')
    r.chunked = True
    r.chunk_left = None
    resp = HTTPResponse('', preload_content=False, headers={'transfer-encoding': 'chunked'}, original_response=r)
    assert resp.chunked is True
    resp.supports_chunked_reads = lambda: True
    resp.release_conn = mock.Mock()
    for _ in resp.stream():
        continue
    resp.release_conn.assert_called_once_with()