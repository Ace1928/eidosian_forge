import socket
import unittest
import httplib2
from six.moves import http_client
from mock import patch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
@unittest.skipIf(not _TOKEN_REFRESH_STATUS_AVAILABLE, 'oauth2client<1.5 lacks HttpAccessTokenRefreshError.')
def testExceptionHandlerHttpAccessTokenError(self):
    exception_arg = HttpAccessTokenRefreshError(status=503)
    retry_args = http_wrapper.ExceptionRetryArgs(http={'connections': {}}, http_request=_MockHttpRequest(), exc=exception_arg, num_retries=0, max_retry_wait=0, total_wait_sec=0)
    with patch('time.sleep', return_value=None):
        http_wrapper.HandleExceptionsAndRebuildHttpConnections(retry_args)