import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
def test_multi_commands(self):
    """Check connection re-use"""
    HTTPretty.register_uri(HTTPretty.POST, 'https://%s/' % self.region.endpoint, json.dumps({'test': 'secure'}), content_type='application/json')
    conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret')
    resp1 = conn.make_request('myCmd1', {'par1': 'foo', 'par2': 'baz'}, '/', 'POST')
    body1 = parse_qs(HTTPretty.last_request.body)
    resp2 = conn.make_request('myCmd2', {'par3': 'bar', 'par4': 'narf'}, '/', 'POST')
    body2 = parse_qs(HTTPretty.last_request.body)
    self.assertEqual(body1[b'par1'], [b'foo'])
    self.assertEqual(body1[b'par2'], [b'baz'])
    with self.assertRaises(KeyError):
        body1[b'par3']
    self.assertEqual(body2[b'par3'], [b'bar'])
    self.assertEqual(body2[b'par4'], [b'narf'])
    with self.assertRaises(KeyError):
        body2['par1']
    self.assertEqual(resp1.read(), b'{"test": "secure"}')
    self.assertEqual(resp2.read(), b'{"test": "secure"}')