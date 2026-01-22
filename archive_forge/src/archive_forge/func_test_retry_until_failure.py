from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
@mock.patch('oslo_db.api.time.sleep', return_value=None)
def test_retry_until_failure(self, p_time_sleep):
    self.dbapi = api.DBAPI('sqlalchemy', {'sqlalchemy': __name__}, use_db_reconnect=True, retry_interval=1, inc_retry_interval=False, max_retries=3)
    func = self.dbapi.api_raise_enable_retry
    self.test_db_api.error_counter = 5
    self.assertRaises(exception.DBError, func, 'Retry of permanent failure did not throw DBError exception.')
    p_time_sleep.assert_called_with(1)
    self.assertNotEqual(0, self.test_db_api.error_counter, 'Retry did not stop after sql_max_retries iterations.')