import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_error_message(self):
    retry = Retry(total=0)
    with pytest.raises(MaxRetryError) as e:
        retry = retry.increment(method='GET', error=ReadTimeoutError(None, '/', 'read timed out'))
    assert 'Caused by redirect' not in str(e.value)
    assert str(e.value.reason) == 'None: read timed out'
    retry = Retry(total=1)
    with pytest.raises(MaxRetryError) as e:
        retry = retry.increment('POST', '/')
        retry = retry.increment('POST', '/')
    assert 'Caused by redirect' not in str(e.value)
    assert isinstance(e.value.reason, ResponseError)
    assert str(e.value.reason) == ResponseError.GENERIC_ERROR
    retry = Retry(total=1)
    response = HTTPResponse(status=500)
    with pytest.raises(MaxRetryError) as e:
        retry = retry.increment('POST', '/', response=response)
        retry = retry.increment('POST', '/', response=response)
    assert 'Caused by redirect' not in str(e.value)
    msg = ResponseError.SPECIFIC_ERROR.format(status_code=500)
    assert str(e.value.reason) == msg
    retry = Retry(connect=1)
    with pytest.raises(MaxRetryError) as e:
        retry = retry.increment(error=ConnectTimeoutError('conntimeout'))
        retry = retry.increment(error=ConnectTimeoutError('conntimeout'))
    assert 'Caused by redirect' not in str(e.value)
    assert str(e.value.reason) == 'conntimeout'