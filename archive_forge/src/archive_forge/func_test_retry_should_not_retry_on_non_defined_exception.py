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
def test_retry_should_not_retry_on_non_defined_exception(self, mock_connect):
    con = Connection()
    con.connection = Mock()
    self.retry_counter = 0
    mock_connect.__name__ = 'mock_connect'
    mock_connect.side_effect = ValueError('should not retry this error')
    retry_request = Retry(timeout=5, retry_delay=0.1, backoff=1)
    self.assertRaisesRegex(ValueError, 'should not retry this error', retry_request(con.request), action='/')
    self.assertEqual(mock_connect.call_count, 1, 'Retry logic failed')