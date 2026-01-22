import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
@mock.patch('oslo_utils.timeutils.now')
def test_splits(self, mock_now):
    mock_now.side_effect = monotonic_iter()
    watch = timeutils.StopWatch()
    watch.start()
    self.assertEqual(0, len(watch.splits))
    watch.split()
    self.assertEqual(1, len(watch.splits))
    self.assertEqual(watch.splits[0].elapsed, watch.splits[0].length)
    watch.split()
    splits = watch.splits
    self.assertEqual(2, len(splits))
    self.assertNotEqual(splits[0].elapsed, splits[1].elapsed)
    self.assertEqual(splits[1].length, splits[1].elapsed - splits[0].elapsed)
    watch.stop()
    self.assertEqual(2, len(watch.splits))
    watch.start()
    self.assertEqual(0, len(watch.splits))