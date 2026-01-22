import os
import sys
import time
import random
import os.path
import platform
import warnings
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import libcloud.security
from libcloud.http import LibcloudConnection
from libcloud.test import unittest, no_network
from libcloud.utils.py3 import reload, httplib, assertRaisesRegex
@unittest.skipIf(no_network(), 'Network is disabled')
def test_prepared_request_empty_body_chunked_encoding_not_used(self):
    connection = LibcloudConnection(host=self.listen_host, port=self.listen_port)
    connection.prepared_request(method='GET', url='/test/prepared-request-1', body='', stream=True)
    self.assertEqual(connection.response.status_code, httplib.OK)
    self.assertEqual(connection.response.content, b'/test/prepared-request-1')
    connection = LibcloudConnection(host=self.listen_host, port=self.listen_port)
    connection.prepared_request(method='GET', url='/test/prepared-request-2', body=None, stream=True)
    self.assertEqual(connection.response.status_code, httplib.OK)
    self.assertEqual(connection.response.content, b'/test/prepared-request-2')