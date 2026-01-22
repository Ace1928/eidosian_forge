import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
@mock.patch('time.sleep')
@mock.patch('oslo_utils.timeutils.now')
def test_timed_custom_message(self, mock_now, mock_sleep):
    mock_now.side_effect = monotonic_iter(incr=0.1)
    fake_logger = mock.MagicMock(logging.getLogger(), autospec=True)

    @timeutils.time_it(fake_logger, message='That took a long time')
    def slow_function():
        time.sleep(0.1)
    slow_function()
    self.assertTrue(mock_now.called)
    self.assertTrue(mock_sleep.called)
    self.assertTrue(fake_logger.log.called)
    fake_logger.log.assert_called_with(logging.DEBUG, 'That took a long time', mock.ANY)