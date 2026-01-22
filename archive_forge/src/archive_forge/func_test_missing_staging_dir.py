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
@mock.patch('os.path.exists')
@mock.patch('glance.sqlite_migration.can_migrate_to_central_db')
def test_missing_staging_dir(self, mock_migrate_db, mock_exists):
    mock_migrate_db.return_value = False
    mock_exists.return_value = False
    server = wsgi.Server()
    with mock.patch.object(server, 'start_wsgi'):
        server.pool = mock.MagicMock()
        with mock.patch.object(wsgi, 'LOG') as mock_log:
            server.start('fake-application', 34567)
            mock_exists.assert_called_once_with('/tmp/staging/')
            mock_log.warning.assert_called_once_with('Import methods are enabled but staging directory %(path)s does not exist; Imports will fail!', {'path': '/tmp/staging/'})