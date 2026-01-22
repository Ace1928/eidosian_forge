import sys
from unittest import mock
from oslo_config import fixture as config_fixture
from oslo_db import concurrency
from oslo_db.tests import base as test_base
@mock.patch('oslo_db.api.DBAPI')
def test_db_api_without_installed_eventlet(self, mock_db_api):
    self.conf.set_override('use_tpool', True, group='database')
    sys.modules['eventlet'] = None
    self.assertRaises(ImportError, getattr, self.db_api, 'fake')