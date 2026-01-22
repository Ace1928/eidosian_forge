import io
import re
import socket
from gunicorn.http.body import ChunkedReader, LengthReader, EOFReader, Body
from gunicorn.http.errors import (
from gunicorn.http.errors import InvalidProxyLine, ForbiddenProxyRequest
from gunicorn.http.errors import InvalidSchemeHeaders
from gunicorn.util import bytes_to_str, split_request_uri
def proxy_protocol_access_check(self):
    if '*' not in self.cfg.proxy_allow_ips and isinstance(self.peer_addr, tuple) and (self.peer_addr[0] not in self.cfg.proxy_allow_ips):
        raise ForbiddenProxyRequest(self.peer_addr[0])