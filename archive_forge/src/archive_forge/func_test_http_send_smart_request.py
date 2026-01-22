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
def test_http_send_smart_request(self):
    post_body = b'hello\n'
    expected_reply_body = b'ok\x012\n'
    http_transport = transport.get_transport_from_url(self.http_server.get_url())
    medium = http_transport.get_smart_medium()
    response = medium.send_http_smart_request(post_body)
    reply_body = response.read()
    self.assertEqual(expected_reply_body, reply_body)