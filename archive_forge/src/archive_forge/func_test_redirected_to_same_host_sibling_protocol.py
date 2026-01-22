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
def test_redirected_to_same_host_sibling_protocol(self):
    t = self._transport('http://www.example.com/foo')
    r = t._redirected_to('http://www.example.com/foo', 'https://www.example.com/foo')
    self.assertIsInstance(r, type(t))
    self.assertEqual('https://www.example.com/foo/', r.external_url())