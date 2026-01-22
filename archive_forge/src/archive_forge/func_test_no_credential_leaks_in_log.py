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
def test_no_credential_leaks_in_log(self):
    self.overrideAttr(debug, 'debug_flags', {'http'})
    user = 'joe'
    password = 'very-sensitive-password'
    self.server.add_user(user, password)
    t = self.get_user_transport(user, password)
    self.mutters = []

    def mutter(*args):
        lines = args[0] % args[1:]
        self.mutters.extend(lines.splitlines())
    self.overrideAttr(trace, 'mutter', mutter)
    self.assertEqual(True, t.has('a'))
    self.assertEqual(1, self.server.auth_required_errors)
    sent_auth_headers = [line for line in self.mutters if line.startswith('> {}'.format(self._auth_header))]
    self.assertLength(1, sent_auth_headers)
    self.assertStartsWith(sent_auth_headers[0], '> {}: <masked>'.format(self._auth_header))