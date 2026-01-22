import io
import socket
import sys
import threading
from http.client import UnknownProtocol, parse_headers
from http.server import SimpleHTTPRequestHandler
import breezy
from .. import (config, controldir, debug, errors, osutils, tests, trace,
from ..bzr import remote as _mod_remote
from ..transport import remote
from ..transport.http import urllib
from ..transport.http.urllib import (AbstractAuthHandler, BasicAuthHandler,
from . import features, http_server, http_utils, test_server
from .scenarios import load_tests_apply_scenarios, multiply_scenarios
def test_smart_http_server_post_request_handler(self):
    httpd = self.http_server.server
    socket = SampleSocket(b'POST /.bzr/smart %s \r\n' % self._protocol_version.encode('ascii') + b'Content-Length: 6\r\n\r\nhello\n')
    request_handler = http_utils.SmartRequestHandler(socket, ('localhost', 80), httpd)
    response = socket.writefile.getvalue()
    self.assertStartsWith(response, b'%s 200 ' % self._protocol_version.encode('ascii'))
    expected_end_of_response = b'\r\n\r\nok\x012\n'
    self.assertEndsWith(response, expected_end_of_response)