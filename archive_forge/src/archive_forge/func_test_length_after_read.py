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
def test_length_after_read(self):
    headers = {'content-length': '5'}
    fp = BytesIO(b'12345')
    resp = HTTPResponse(fp, preload_content=False)
    resp.read()
    assert resp.length_remaining is None
    fp = BytesIO(b'12345')
    resp = HTTPResponse(fp, headers=headers, preload_content=False)
    resp.read()
    assert resp.length_remaining == 0
    fp = BytesIO(b'12345')
    resp = HTTPResponse(fp, headers=headers, preload_content=False)
    data = resp.stream(2)
    next(data)
    assert resp.length_remaining == 3