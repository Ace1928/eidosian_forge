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
def test_open_controldir(self):
    branch = self.make_branch('relpath')
    url = self.http_server.get_url() + 'relpath'
    bd = controldir.ControlDir.open(url)
    self.addCleanup(bd.transport.disconnect)
    self.assertIsInstance(bd, _mod_remote.RemoteBzrDir)