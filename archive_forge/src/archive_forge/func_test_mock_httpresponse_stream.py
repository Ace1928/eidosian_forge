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
def test_mock_httpresponse_stream(self):

    class MockHTTPRequest(object):
        self.fp = None

        def read(self, amt):
            data = self.fp.read(amt)
            if not data:
                self.fp = None
            return data

        def close(self):
            self.fp = None
    bio = BytesIO(b'foo')
    fp = MockHTTPRequest()
    fp.fp = bio
    resp = HTTPResponse(fp, preload_content=False)
    stream = resp.stream(2)
    assert next(stream) == b'fo'
    assert next(stream) == b'o'
    with pytest.raises(StopIteration):
        next(stream)