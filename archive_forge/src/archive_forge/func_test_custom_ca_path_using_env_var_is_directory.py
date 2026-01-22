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
def test_custom_ca_path_using_env_var_is_directory(self):
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.environ['SSL_CERT_FILE'] = file_path
    expected_msg = "Certificate file can't be a directory"
    assertRaisesRegex(self, ValueError, expected_msg, reload, libcloud.security)