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
def test_length_w_bad_header(self):
    garbage = {'content-length': 'foo'}
    fp = BytesIO(b'12345')
    resp = HTTPResponse(fp, headers=garbage, preload_content=False)
    assert resp.length_remaining is None
    garbage['content-length'] = '-10'
    resp = HTTPResponse(fp, headers=garbage, preload_content=False)
    assert resp.length_remaining is None