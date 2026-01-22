import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_offset_west(self):
    time_str = '2012-02-14T20:53:07-05:30'
    offset = -5.5 * 60 * 60
    self._do_test(time_str, 2012, 2, 14, 20, 53, 7, 0, offset)