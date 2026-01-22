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
def test_no_timed_disabled(self, mock_now, mock_sleep):
    mock_now.side_effect = monotonic_iter(incr=0.1)
    fake_logger = mock.MagicMock(logging.getLogger(), autospec=True)

    @timeutils.time_it(fake_logger, enabled=False)
    def slow_function():
        time.sleep(0.1)
    slow_function()
    self.assertFalse(mock_now.called)
    self.assertFalse(fake_logger.log.called)