import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
@mock.patch('oslo_utils.timeutils.now')
def test_not_expired(self, mock_now):
    mock_now.side_effect = monotonic_iter()
    watch = timeutils.StopWatch(0.1)
    watch.start()
    self.assertFalse(watch.expired())