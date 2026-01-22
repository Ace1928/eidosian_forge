import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_retry_set_remove_headers_on_redirect(self):
    retry = Retry(remove_headers_on_redirect=['X-API-Secret'])
    assert list(retry.remove_headers_on_redirect) == ['x-api-secret']