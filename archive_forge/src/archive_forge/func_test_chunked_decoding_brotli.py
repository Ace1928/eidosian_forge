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
@onlyBrotlipy()
def test_chunked_decoding_brotli(self):
    data = brotli.compress(b'foobarbaz')
    fp = BytesIO(data)
    r = HTTPResponse(fp, headers={'content-encoding': 'br'}, preload_content=False)
    ret = b''
    for _ in range(100):
        ret += r.read(1)
        if r.closed:
            break
    assert ret == b'foobarbaz'