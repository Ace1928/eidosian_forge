import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_has_started_stopped(self):
    watch = timeutils.StopWatch()
    self.assertFalse(watch.has_started())
    self.assertFalse(watch.has_stopped())
    watch.start()
    self.assertTrue(watch.has_started())
    self.assertFalse(watch.has_stopped())
    watch.stop()
    self.assertTrue(watch.has_stopped())
    self.assertFalse(watch.has_started())