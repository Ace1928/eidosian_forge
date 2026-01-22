import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
def test_build_base_http_request_noproxy(self):
    os.environ['no_proxy'] = 'mockservice.cc-zone-1.amazonaws.com'
    conn = AWSAuthConnection('mockservice.cc-zone-1.amazonaws.com', aws_access_key_id='access_key', aws_secret_access_key='secret', suppress_consec_slashes=False, proxy='127.0.0.1', proxy_user='john.doe', proxy_pass='p4ssw0rd', proxy_port='8180')
    request = conn.build_base_http_request('GET', '/', None)
    del os.environ['no_proxy']
    self.assertEqual(request.path, '/')