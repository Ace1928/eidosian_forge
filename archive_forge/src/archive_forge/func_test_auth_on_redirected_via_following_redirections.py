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
def test_auth_on_redirected_via_following_redirections(self):
    self.new_server.add_user('joe', 'foo')
    ui.ui_factory = tests.TestUIFactory(stdin='joe\nfoo\n')
    t = self.old_transport
    new_prefix = 'http://{}:{}'.format(self.new_server.host, self.new_server.port)
    self.old_server.redirections = [('(.*)', '%s/1\\1' % new_prefix, 301)]
    self.assertEqual(b'redirected once', t.request('GET', t.abspath('a'), retries=3).read())
    self.assertEqual('', ui.ui_factory.stdin.readline())
    self.assertEqual('', ui.ui_factory.stdout.getvalue())