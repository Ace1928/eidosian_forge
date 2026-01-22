import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_is_older_than_aware(self):
    """Tests sending is_older_than an 'aware' datetime."""
    self._test_is_older_than(lambda x: x.replace(tzinfo=iso8601.iso8601.UTC))