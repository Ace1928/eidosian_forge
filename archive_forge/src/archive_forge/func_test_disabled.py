import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
@pytest.mark.parametrize('total', [-1, 0])
def test_disabled(self, total):
    with pytest.raises(MaxRetryError):
        Retry(total).increment(method='GET')