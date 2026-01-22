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
def cleanup_http_redirection_connections(test):

    def socket_disconnect(sock):
        try:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        except OSError:
            pass

    def connect(connection):
        test.http_connect_orig(connection)
        test.addCleanup(socket_disconnect, connection.sock)
    test.http_connect_orig = test.overrideAttr(HTTPConnection, 'connect', connect)

    def connect(connection):
        test.https_connect_orig(connection)
        test.addCleanup(socket_disconnect, connection.sock)
    test.https_connect_orig = test.overrideAttr(HTTPSConnection, 'connect', connect)