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
def test_implicit_port(self):
    """
        Test that the port is not included in the URL if the protocol implies
        the port, e.g. http implies 80
        """
    conn = Connection(secure=True, host='localhost', port=443)
    conn.connect()
    self.assertEqual(conn.connection.host, 'https://localhost')
    conn2 = Connection(secure=False, host='localhost', port=80)
    conn2.connect()
    self.assertEqual(conn2.connection.host, 'http://localhost')