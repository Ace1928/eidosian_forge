import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_status_counter(self):
    resp = HTTPResponse(status=400)
    retry = Retry(status=2)
    retry = retry.increment(response=resp)
    retry = retry.increment(response=resp)
    with pytest.raises(MaxRetryError) as e:
        retry.increment(response=resp)
    assert str(e.value.reason) == ResponseError.SPECIFIC_ERROR.format(status_code=400)