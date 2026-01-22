import sys
from unittest import mock
from oslo_config import fixture as config_fixture
from oslo_db import concurrency
from oslo_db.tests import base as test_base
@mock.patch('oslo_db.api.DBAPI')
def test_db_api_common(self, mock_db_api):
    fake_db_api = mock.MagicMock()
    mock_db_api.from_config.return_value = fake_db_api
    self.db_api.fake_call_1
    mock_db_api.from_config.assert_called_once_with(conf=self.conf, backend_mapping=FAKE_BACKEND_MAPPING)
    self.assertEqual(fake_db_api, self.db_api._db_api)
    self.assertFalse(self.eventlet.tpool.Proxy.called)
    self.db_api.fake_call_2
    self.assertEqual(fake_db_api, self.db_api._db_api)
    self.assertFalse(self.eventlet.tpool.Proxy.called)
    self.assertEqual(1, mock_db_api.from_config.call_count)