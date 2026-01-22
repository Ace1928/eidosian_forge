import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_sleep(self):
    retry = Retry(backoff_factor=0.0001)
    retry = retry.increment(method='GET')
    retry = retry.increment(method='GET')
    retry.sleep()