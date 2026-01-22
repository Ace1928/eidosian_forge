import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_backoff(self):
    """Backoff is computed correctly"""
    max_backoff = Retry.DEFAULT_BACKOFF_MAX
    retry = Retry(total=100, backoff_factor=0.2)
    assert retry.get_backoff_time() == 0
    retry = retry.increment(method='GET')
    assert retry.get_backoff_time() == 0
    retry = retry.increment(method='GET')
    assert retry.backoff_factor == 0.2
    assert retry.total == 98
    assert retry.get_backoff_time() == 0.4
    retry = retry.increment(method='GET')
    assert retry.get_backoff_time() == 0.8
    retry = retry.increment(method='GET')
    assert retry.get_backoff_time() == 1.6
    for _ in xrange(10):
        retry = retry.increment(method='GET')
    assert retry.get_backoff_time() == max_backoff