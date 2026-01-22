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
def test_post(self):
    self.server.canned_response = b'HTTP/1.1 200 OK\r\nDate: Tue, 11 Jul 2006 04:32:56 GMT\r\nServer: Apache/2.0.54 (Fedora)\r\nLast-Modified: Sun, 23 Apr 2006 19:35:20 GMT\r\nETag: "56691-23-38e9ae00"\r\nAccept-Ranges: bytes\r\nContent-Length: 35\r\nConnection: close\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\nlalala whatever as long as itsssss\n'
    t = self.get_transport()
    code, f = t._post(b'abc def end-of-body\n')
    self.assertEqual(b'lalala whatever as long as itsssss\n', f.read())
    self.assertActivitiesMatch()