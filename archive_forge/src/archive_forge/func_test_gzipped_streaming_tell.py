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
def test_gzipped_streaming_tell(self):
    compress = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    uncompressed_data = b'foo'
    data = compress.compress(uncompressed_data)
    data += compress.flush()
    fp = BytesIO(data)
    resp = HTTPResponse(fp, headers={'content-encoding': 'gzip'}, preload_content=False)
    stream = resp.stream()
    payload = next(stream)
    assert payload == uncompressed_data
    assert len(data) == resp.tell()
    with pytest.raises(StopIteration):
        next(stream)