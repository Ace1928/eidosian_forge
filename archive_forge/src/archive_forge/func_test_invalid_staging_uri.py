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
@mock.patch('glance.sqlite_migration.can_migrate_to_central_db')
def test_invalid_staging_uri(self, mock_migrate_db):
    mock_migrate_db.return_value = False
    self.config(node_staging_uri='http://good.luck')
    server = wsgi.Server()
    with mock.patch.object(server, 'start_wsgi'):
        self.assertRaises(exception.GlanceException, server.start, 'fake-application', 34567)