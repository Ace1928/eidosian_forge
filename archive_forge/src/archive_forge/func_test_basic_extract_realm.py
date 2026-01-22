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
def test_basic_extract_realm(self):
    scheme, remainder = self.parse_header('Basic realm="Thou should not pass"', BasicAuthHandler)
    match, realm = self.auth_handler.extract_realm(remainder)
    self.assertTrue(match is not None)
    self.assertEqual('Thou should not pass', realm)