import os
import ssl
import sys
import socket
from unittest import mock
from unittest.mock import Mock, patch
import requests_mock
from requests.exceptions import ConnectTimeout
import libcloud.common.base
from libcloud.http import LibcloudConnection, SignedHTTPSAdapter, LibcloudBaseConnection
from libcloud.test import unittest, no_internet
from libcloud.utils.py3 import assertRaisesRegex
from libcloud.common.base import Response, Connection, CertificateConnection
from libcloud.utils.retry import RETRY_EXCEPTIONS, Retry, RetryForeverOnRateLimitError
from libcloud.common.exceptions import RateLimitReachedError
@patch('libcloud.common.base.Connection.request')
def test_retry_rate_limit_error_success_on_second_attempt(self, mock_connect):
    con = Connection()
    con.connection = Mock()
    self.retry_counter = 0

    def mock_connect_side_effect(*args, **kwargs):
        self.retry_counter += 1
        if self.retry_counter < 2:
            headers = {'retry-after': 0.2}
            raise RateLimitReachedError(headers=headers)
        return 'success'
    mock_connect.__name__ = 'mock_connect'
    mock_connect.side_effect = mock_connect_side_effect
    retry_request = Retry(timeout=1, retry_delay=0.1, backoff=1)
    result = retry_request(con.request)(action='/')
    self.assertEqual(result, 'success')
    self.assertEqual(mock_connect.call_count, 2, 'Retry logic failed')