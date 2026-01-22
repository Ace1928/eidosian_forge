import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_retry_default(self):
    """If no value is specified, should retry connects 3 times"""
    retry = Retry()
    assert retry.total == 10
    assert retry.connect is None
    assert retry.read is None
    assert retry.redirect is None
    assert retry.other is None
    error = ConnectTimeoutError()
    retry = Retry(connect=1)
    retry = retry.increment(error=error)
    with pytest.raises(MaxRetryError):
        retry.increment(error=error)
    retry = Retry(connect=1)
    retry = retry.increment(error=error)
    assert not retry.is_exhausted()
    assert Retry(0).raise_on_redirect
    assert not Retry(False).raise_on_redirect