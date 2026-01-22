import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_splits_never_started(self):
    watch = timeutils.StopWatch()
    self.assertRaises(RuntimeError, watch.split)