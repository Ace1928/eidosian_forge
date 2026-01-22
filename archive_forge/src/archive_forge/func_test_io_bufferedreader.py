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
def test_io_bufferedreader(self):
    fp = BytesIO(b'foo')
    resp = HTTPResponse(fp, preload_content=False)
    br = BufferedReader(resp)
    assert br.read() == b'foo'
    br.close()
    assert resp.closed
    fp = BytesIO(b'hello\nworld')
    resp = HTTPResponse(fp, preload_content=False)
    with pytest.raises(ValueError) as ctx:
        list(BufferedReader(resp))
    assert str(ctx.value) == 'readline of closed file'
    b = b'fooandahalf'
    fp = BytesIO(b)
    resp = HTTPResponse(fp, preload_content=False)
    br = BufferedReader(resp, 5)
    br.read(1)
    assert len(fp.read()) == len(b) - 5
    while not br.closed:
        br.read(5)