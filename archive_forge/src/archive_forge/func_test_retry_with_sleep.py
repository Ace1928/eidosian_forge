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
def test_retry_with_sleep(self, mock_connect):
    con = Connection()
    con.connection = Mock()
    mock_connect.side_effect = socket.gaierror('')
    retry_request = Retry(timeout=1, retry_delay=0.1, backoff=1)
    self.assertRaises(socket.gaierror, retry_request(con.request), action='/')
    self.assertGreater(mock_connect.call_count, 1, 'Retry logic failed')