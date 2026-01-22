import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_normalize_naive(self):
    dt = datetime.datetime(2011, 2, 14, 20, 53, 7)
    dtn = datetime.datetime(2011, 2, 14, 19, 53, 7)
    naive = timeutils.normalize_time(dtn)
    self.assertTrue(naive < dt)