import os
import sys
import zlib
from io import StringIO
from unittest import mock
import requests_mock
import libcloud
from libcloud.http import LibcloudConnection
from libcloud.test import unittest
from libcloud.common.base import Connection
from libcloud.utils.loggingconnection import LoggingConnection
def test_log_response_xml_content_type(self):
    conn = LoggingConnection(host='example.com', port=80)
    r = self._get_mock_response('text/xml', '<foo><bar /></foo>')
    result = conn._log_response(r).replace('\r', '')
    self.assertTrue(EXPECTED_DATA_XML in result)