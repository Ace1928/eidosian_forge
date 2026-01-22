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
def test_io_closed_consistently(self, sock):
    try:
        hlr = httplib.HTTPResponse(sock)
        hlr.fp = BytesIO(b'foo')
        hlr.chunked = 0
        hlr.length = 3
        with HTTPResponse(hlr, preload_content=False) as resp:
            assert not resp.closed
            assert not resp._fp.isclosed()
            assert not is_fp_closed(resp._fp)
            assert not resp.isclosed()
            resp.read()
            assert resp.closed
            assert resp._fp.isclosed()
            assert is_fp_closed(resp._fp)
            assert resp.isclosed()
    finally:
        hlr.close()