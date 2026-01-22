import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
@pytest.mark.parametrize('value', ['-1', '+1', '1.0', six.u('²')])
def test_parse_retry_after_invalid(self, value):
    retry = Retry()
    with pytest.raises(InvalidHeader):
        retry.parse_retry_after(value)