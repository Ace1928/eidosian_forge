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
def test_log_response_with_pretty_print_json_content_type(self):
    os.environ['LIBCLOUD_DEBUG_PRETTY_PRINT_RESPONSE'] = '1'
    conn = LoggingConnection(host='example.com', port=80)
    r = self._get_mock_response('application/json', '{"foo": "bar!"}')
    result = conn._log_response(r).replace('\r', '')
    self.assertTrue(EXPECTED_DATA_JSON_PRETTY in result)
    r = self._get_mock_response('application/json', bytes('{"foo": "bar!"}', 'utf-8'))
    result = conn._log_response(r).replace('\r', '')
    self.assertTrue(EXPECTED_DATA_JSON_PRETTY in result)