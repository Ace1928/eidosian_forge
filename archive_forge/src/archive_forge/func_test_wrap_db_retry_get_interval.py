from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
def test_wrap_db_retry_get_interval(self):
    x = api.wrap_db_retry(max_retries=5, retry_on_deadlock=True, max_retry_interval=11)
    self.assertEqual(11, x.max_retry_interval)
    for i in (1, 2, 4):
        sleep_time, n = x._get_inc_interval(i, True)
        self.assertEqual(2 * i, n)
        self.assertTrue(2 * i > sleep_time)
        sleep_time, n = x._get_inc_interval(i, False)
        self.assertEqual(2 * i, n)
        self.assertEqual(2 * i, sleep_time)
    for i in (8, 16, 32):
        sleep_time, n = x._get_inc_interval(i, False)
        self.assertEqual(x.max_retry_interval, sleep_time)
        self.assertEqual(2 * i, n)
        sleep_time, n = x._get_inc_interval(i, True)
        self.assertTrue(x.max_retry_interval >= sleep_time)
        self.assertEqual(2 * i, n)