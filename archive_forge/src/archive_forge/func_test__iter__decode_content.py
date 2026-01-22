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
def test__iter__decode_content(self):

    def stream():
        compress = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
        data = compress.compress(b'foo\nbar')
        data += compress.flush()
        for i in range(0, len(data), 2):
            yield data[i:i + 2]
    fp = MockChunkedEncodingResponse(list(stream()))
    r = httplib.HTTPResponse(MockSock)
    r.fp = fp
    headers = {'transfer-encoding': 'chunked', 'content-encoding': 'gzip'}
    resp = HTTPResponse(r, preload_content=False, headers=headers)
    data = b''
    for c in resp:
        data += c
    assert b'foo\nbar' == data