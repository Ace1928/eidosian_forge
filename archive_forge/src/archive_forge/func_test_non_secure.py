import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
def test_non_secure(self):
    HTTPretty.register_uri(HTTPretty.POST, 'http://%s/' % self.region.endpoint, json.dumps({'test': 'normal'}), content_type='application/json')
    conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret', is_secure=False)
    resp = conn.make_request('myCmd1', {'par1': 'foo', 'par2': 'baz'}, '/', 'POST')
    self.assertEqual(resp.read(), b'{"test": "normal"}')