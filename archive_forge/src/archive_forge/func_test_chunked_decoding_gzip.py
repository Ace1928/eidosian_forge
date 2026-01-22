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
def test_chunked_decoding_gzip(self):
    compress = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    data = compress.compress(b'foo')
    data += compress.flush()
    fp = BytesIO(data)
    r = HTTPResponse(fp, headers={'content-encoding': 'gzip'}, preload_content=False)
    assert r.read(11) == b''
    assert r.read(1) == b'f'
    assert r.read(2) == b'oo'
    assert r.read() == b''
    assert r.read() == b''