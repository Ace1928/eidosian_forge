import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_retry_method_not_in_whitelist(self):
    error = ReadTimeoutError(None, '/', 'read timed out')
    retry = Retry()
    with pytest.raises(ReadTimeoutError):
        retry.increment(method='POST', error=error)