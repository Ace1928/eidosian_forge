import os
import platform
import socket
import tempfile
import testtools
from unittest import mock
import eventlet
import eventlet.wsgi
import requests
import webob
from oslo_config import cfg
from oslo_service import sslutils
from oslo_service.tests import base
from oslo_service import wsgi
from oslo_utils import netutils
@testtools.skipIf(not hasattr(socket, 'AF_UNIX'), 'UNIX sockets not supported')
def test_server_with_unix_socket(self):
    socket_file = self.get_temp_file_path('sock')
    socket_mode = 420
    server = wsgi.Server(self.conf, 'test_socket_options', None, socket_family=socket.AF_UNIX, socket_mode=socket_mode, socket_file=socket_file)
    self.assertEqual(socket_file, server.socket.getsockname())
    self.assertEqual(socket_mode, os.stat(socket_file).st_mode & 511)
    server.start()
    self.assertFalse(server._server.dead)
    server.stop()
    server.wait()
    self.assertTrue(server._server.dead)