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
def test_parse_proxy_url(self):
    conn = LibcloudBaseConnection()
    proxy_url = 'http://127.0.0.1:3128'
    result = conn._parse_proxy_url(proxy_url=proxy_url)
    self.assertEqual(result[0], 'http')
    self.assertEqual(result[1], '127.0.0.1')
    self.assertEqual(result[2], 3128)
    self.assertIsNone(result[3])
    self.assertIsNone(result[4])
    proxy_url = 'https://127.0.0.2:3129'
    result = conn._parse_proxy_url(proxy_url=proxy_url)
    self.assertEqual(result[0], 'https')
    self.assertEqual(result[1], '127.0.0.2')
    self.assertEqual(result[2], 3129)
    self.assertIsNone(result[3])
    self.assertIsNone(result[4])
    proxy_url = 'http://user1:pass1@127.0.0.1:3128'
    result = conn._parse_proxy_url(proxy_url=proxy_url)
    self.assertEqual(result[0], 'http')
    self.assertEqual(result[1], '127.0.0.1')
    self.assertEqual(result[2], 3128)
    self.assertEqual(result[3], 'user1')
    self.assertEqual(result[4], 'pass1')
    proxy_url = 'https://user1:pass1@127.0.0.2:3129'
    result = conn._parse_proxy_url(proxy_url=proxy_url)
    self.assertEqual(result[0], 'https')
    self.assertEqual(result[1], '127.0.0.2')
    self.assertEqual(result[2], 3129)
    self.assertEqual(result[3], 'user1')
    self.assertEqual(result[4], 'pass1')
    proxy_url = 'http://127.0.0.1'
    expected_msg = 'proxy_url must be in the following format'
    assertRaisesRegex(self, ValueError, expected_msg, conn._parse_proxy_url, proxy_url=proxy_url)
    proxy_url = 'http://@127.0.0.1:3128'
    expected_msg = 'URL is in an invalid format'
    assertRaisesRegex(self, ValueError, expected_msg, conn._parse_proxy_url, proxy_url=proxy_url)
    proxy_url = 'http://user@127.0.0.1:3128'
    expected_msg = 'URL is in an invalid format'
    assertRaisesRegex(self, ValueError, expected_msg, conn._parse_proxy_url, proxy_url=proxy_url)