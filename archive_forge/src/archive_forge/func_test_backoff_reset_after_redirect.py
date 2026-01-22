import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_backoff_reset_after_redirect(self):
    retry = Retry(total=100, redirect=5, backoff_factor=0.2)
    retry = retry.increment(method='GET')
    retry = retry.increment(method='GET')
    assert retry.get_backoff_time() == 0.4
    redirect_response = HTTPResponse(status=302, headers={'location': 'test'})
    retry = retry.increment(method='GET', response=redirect_response)
    assert retry.get_backoff_time() == 0
    retry = retry.increment(method='GET')
    retry = retry.increment(method='GET')
    assert retry.get_backoff_time() == 0.4