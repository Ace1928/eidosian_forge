import datetime
import gettext
import http.client as http
import os
import socket
from unittest import mock
import eventlet.patcher
import fixtures
from oslo_concurrency import processutils
from oslo_serialization import jsonutils
import routes
import webob
from glance.api.v2 import router as router_v2
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance import i18n
from glance.image_cache import prefetcher
from glance.tests import utils as test_utils
@mock.patch.object(prefetcher, 'Prefetcher')
@mock.patch.object(wsgi.Server, 'configure_socket')
@mock.patch('glance.sqlite_migration.can_migrate_to_central_db')
def test_http_keepalive(self, mock_migrate_db, mock_configure_socket, mock_prefetcher):
    mock_migrate_db.return_value = False
    self.config(http_keepalive=False)
    self.config(workers=0)
    server = wsgi.Server(threads=1)
    server.sock = 'fake_socket'
    with mock.patch.object(eventlet.wsgi, 'server') as mock_server:
        fake_application = 'fake-application'
        server.start(fake_application, 0)
        server.wait()
        mock_server.assert_called_once_with('fake_socket', fake_application, log=server._logger, debug=False, custom_pool=server.pool, keepalive=False, socket_timeout=900)