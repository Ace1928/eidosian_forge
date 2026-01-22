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
def test_changing_nonce(self):
    if self._auth_server not in (http_utils.HTTPDigestAuthServer, http_utils.ProxyDigestAuthServer):
        raise tests.TestNotApplicable('HTTP/proxy auth digest only test')
    self.server.add_user('joe', 'foo')
    t = self.get_user_transport('joe', 'foo')
    with t.get('a') as f:
        self.assertEqual(b'contents of a\n', f.read())
    with t.get('b') as f:
        self.assertEqual(b'contents of b\n', f.read())
    self.assertEqual(1, self.server.auth_required_errors)
    self.server.auth_nonce = self.server.auth_nonce + '. No, now!'
    self.assertEqual(b'contents of a\n', t.get('a').read())
    self.assertEqual(2, self.server.auth_required_errors)