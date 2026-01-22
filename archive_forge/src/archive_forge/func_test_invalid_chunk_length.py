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
def test_invalid_chunk_length(self):
    stream = [b'foooo', b'bbbbaaaaar']
    fp = MockChunkedInvalidChunkLength(stream)
    r = httplib.HTTPResponse(MockSock)
    r.fp = fp
    r.chunked = True
    r.chunk_left = None
    resp = HTTPResponse(r, preload_content=False, headers={'transfer-encoding': 'chunked'})
    with pytest.raises(ProtocolError) as ctx:
        next(resp.read_chunked())
    orig_ex = ctx.value.args[1]
    assert isinstance(orig_ex, InvalidChunkLength)
    assert orig_ex.length == six.b(fp.BAD_LENGTH_LINE)