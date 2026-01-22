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
def test_incomplete_readv_leave_pipe_clean(self):
    server = self.get_readonly_server()
    t = self.get_readonly_transport()
    t._get_max_size = 2
    ireadv = iter(t.readv('a', ((0, 1), (1, 1), (2, 4), (6, 4))))
    self.assertEqual((0, b'0'), next(ireadv))
    self.assertEqual(1, server.GET_request_nb)
    self.assertEqual(b'0123456789', t.get_bytes('a'))
    self.assertEqual(2, server.GET_request_nb)