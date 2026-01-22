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
def test_deflate2_streaming(self):
    compress = zlib.compressobj(6, zlib.DEFLATED, -zlib.MAX_WBITS)
    data = compress.compress(b'foo')
    data += compress.flush()
    fp = BytesIO(data)
    resp = HTTPResponse(fp, headers={'content-encoding': 'deflate'}, preload_content=False)
    stream = resp.stream(2)
    assert next(stream) == b'f'
    assert next(stream) == b'oo'
    with pytest.raises(StopIteration):
        next(stream)