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
def test_wsgi_keep_alive(self):
    self.config(wsgi_keep_alive=False)
    with mock.patch.object(eventlet, 'spawn') as mock_spawn:
        server = wsgi.Server(self.conf, 'test_app', None, host='127.0.0.1', port=0)
        server.start()
        _, kwargs = mock_spawn.call_args
        self.assertEqual(self.conf.wsgi_keep_alive, kwargs['keepalive'])
        server.stop()