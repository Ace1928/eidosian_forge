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
def test_request_custom_timeout_timeout(self):

    def response_hook(*args, **kwargs):
        self.assertEqual(kwargs['timeout'], 0.5)
    hooks = {'response': response_hook}
    connection = LibcloudConnection(host=self.listen_host, port=self.listen_port, timeout=0.5)
    self.assertRaisesRegex(requests.exceptions.ReadTimeout, 'Read timed out', connection.request, method='GET', url='/test-timeout', hooks=hooks)