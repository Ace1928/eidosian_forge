import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
@mock.patch('oslo_utils.timeutils.now')
def test_pause_resume(self, mock_now):
    mock_now.side_effect = monotonic_iter()
    watch = timeutils.StopWatch()
    watch.start()
    watch.stop()
    elapsed = watch.elapsed()
    self.assertAlmostEqual(elapsed, watch.elapsed())
    watch.resume()
    self.assertNotEqual(elapsed, watch.elapsed())