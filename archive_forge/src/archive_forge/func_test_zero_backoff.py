import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_zero_backoff(self):
    retry = Retry()
    assert retry.get_backoff_time() == 0
    retry = retry.increment(method='GET')
    retry = retry.increment(method='GET')
    assert retry.get_backoff_time() == 0