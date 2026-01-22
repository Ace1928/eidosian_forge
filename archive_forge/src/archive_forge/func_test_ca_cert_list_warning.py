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
def test_ca_cert_list_warning(self):
    with warnings.catch_warnings(record=True) as w:
        self.httplib_object.verify = True
        self.httplib_object._setup_ca_cert(ca_cert=[ORIGINAL_CA_CERTS_PATH])
        self.assertEqual(self.httplib_object.ca_cert, ORIGINAL_CA_CERTS_PATH)
        self.assertEqual(w[0].category, DeprecationWarning)