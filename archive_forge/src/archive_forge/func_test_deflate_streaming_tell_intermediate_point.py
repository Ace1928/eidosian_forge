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
def test_deflate_streaming_tell_intermediate_point(self):
    NUMBER_OF_READS = 10

    class MockCompressedDataReading(BytesIO):
        """
            A BytesIO-like reader returning ``payload`` in ``NUMBER_OF_READS``
            calls to ``read``.
            """

        def __init__(self, payload, payload_part_size):
            self.payloads = [payload[i * payload_part_size:(i + 1) * payload_part_size] for i in range(NUMBER_OF_READS + 1)]
            assert b''.join(self.payloads) == payload

        def read(self, _):
            if len(self.payloads) > 0:
                return self.payloads.pop(0)
            return b''
    uncompressed_data = zlib.decompress(ZLIB_PAYLOAD)
    payload_part_size = len(ZLIB_PAYLOAD) // NUMBER_OF_READS
    fp = MockCompressedDataReading(ZLIB_PAYLOAD, payload_part_size)
    resp = HTTPResponse(fp, headers={'content-encoding': 'deflate'}, preload_content=False)
    stream = resp.stream()
    parts_positions = [(part, resp.tell()) for part in stream]
    end_of_stream = resp.tell()
    with pytest.raises(StopIteration):
        next(stream)
    parts, positions = zip(*parts_positions)
    payload = b''.join(parts)
    assert uncompressed_data == payload
    expected = [(i + 1) * payload_part_size for i in range(NUMBER_OF_READS)]
    assert expected == list(positions)
    assert len(ZLIB_PAYLOAD) == end_of_stream