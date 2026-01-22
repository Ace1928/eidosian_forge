import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
def test_host_header_with_nonstandard_port(self):
    conn = V4AuthConnection('testhost', aws_access_key_id='access_key', aws_secret_access_key='secret')
    request = conn.build_base_http_request(method='POST', path='/', auth_path=None, params=None, headers=None, data='', host=None)
    conn.set_host_header(request)
    self.assertEqual(request.headers['Host'], 'testhost')
    conn = V4AuthConnection('testhost', aws_access_key_id='access_key', aws_secret_access_key='secret', port=8773)
    request = conn.build_base_http_request(method='POST', path='/', auth_path=None, params=None, headers=None, data='', host=None)
    conn.set_host_header(request)
    self.assertEqual(request.headers['Host'], 'testhost:8773')