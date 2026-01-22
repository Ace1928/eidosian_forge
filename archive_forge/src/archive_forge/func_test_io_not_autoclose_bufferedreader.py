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
def test_io_not_autoclose_bufferedreader(self):
    fp = BytesIO(b'hello\nworld')
    resp = HTTPResponse(fp, preload_content=False, auto_close=False)
    reader = BufferedReader(resp)
    assert list(reader) == [b'hello\n', b'world']
    assert not reader.closed
    assert not resp.closed
    with pytest.raises(StopIteration):
        next(reader)
    reader.close()
    assert reader.closed
    assert resp.closed
    with pytest.raises(ValueError) as ctx:
        next(reader)
    assert str(ctx.value) == 'readline of closed file'