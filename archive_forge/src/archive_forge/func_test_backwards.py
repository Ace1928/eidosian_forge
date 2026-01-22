import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
@mock.patch('oslo_utils.timeutils.now')
def test_backwards(self, mock_now):
    mock_now.side_effect = [0, 0.5, -1.0, -1.0]
    watch = timeutils.StopWatch(0.1)
    watch.start()
    self.assertTrue(watch.expired())
    self.assertFalse(watch.expired())
    self.assertEqual(0.0, watch.elapsed())