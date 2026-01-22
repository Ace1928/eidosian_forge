import io
import re
import socket
from gunicorn.http.body import ChunkedReader, LengthReader, EOFReader, Body
from gunicorn.http.errors import (
from gunicorn.http.errors import InvalidProxyLine, ForbiddenProxyRequest
from gunicorn.http.errors import InvalidSchemeHeaders
from gunicorn.util import bytes_to_str, split_request_uri
def parse_request_line(self, line_bytes):
    bits = [bytes_to_str(bit) for bit in line_bytes.split(None, 2)]
    if len(bits) != 3:
        raise InvalidRequestLine(bytes_to_str(line_bytes))
    if not METH_RE.match(bits[0]):
        raise InvalidRequestMethod(bits[0])
    self.method = bits[0].upper()
    self.uri = bits[1]
    try:
        parts = split_request_uri(self.uri)
    except ValueError:
        raise InvalidRequestLine(bytes_to_str(line_bytes))
    self.path = parts.path or ''
    self.query = parts.query or ''
    self.fragment = parts.fragment or ''
    match = VERSION_RE.match(bits[2])
    if match is None:
        raise InvalidHTTPVersion(bits[2])
    self.version = (int(match.group(1)), int(match.group(2)))