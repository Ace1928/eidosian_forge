import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_cls_set_default_method_whitelist(self, expect_retry_deprecation):
    old_setting = Retry.DEFAULT_METHOD_WHITELIST
    try:
        Retry.DEFAULT_METHOD_WHITELIST = {'GET'}
        retry = Retry()
        assert retry.DEFAULT_ALLOWED_METHODS == {'GET'}
        assert retry.DEFAULT_METHOD_WHITELIST == {'GET'}
        assert retry.allowed_methods == {'GET'}
        assert retry.method_whitelist == {'GET'}
        retry = Retry(allowed_methods={'GET', 'POST'})
        assert retry.DEFAULT_ALLOWED_METHODS == {'GET'}
        assert retry.DEFAULT_METHOD_WHITELIST == {'GET'}
        assert retry.allowed_methods == {'GET', 'POST'}
        assert retry.method_whitelist == {'GET', 'POST'}
        retry = Retry(method_whitelist={'POST'})
        assert retry.DEFAULT_ALLOWED_METHODS == {'GET'}
        assert retry.DEFAULT_METHOD_WHITELIST == {'GET'}
        assert retry.allowed_methods == {'POST'}
        assert retry.method_whitelist == {'POST'}
    finally:
        Retry.DEFAULT_METHOD_WHITELIST = old_setting
        assert Retry.DEFAULT_ALLOWED_METHODS == old_setting