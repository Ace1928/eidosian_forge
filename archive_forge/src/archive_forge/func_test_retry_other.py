import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_retry_other(self):
    """If an unexpected error is raised, should retry other times"""
    other_error = SSLError()
    retry = Retry(connect=1)
    retry = retry.increment(error=other_error)
    retry = retry.increment(error=other_error)
    assert not retry.is_exhausted()
    retry = Retry(other=1)
    retry = retry.increment(error=other_error)
    with pytest.raises(MaxRetryError) as e:
        retry.increment(error=other_error)
    assert e.value.reason == other_error