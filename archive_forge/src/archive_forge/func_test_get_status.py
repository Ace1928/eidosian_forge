import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
def test_get_status(self):
    HTTPretty.register_uri(HTTPretty.GET, 'https://%s/status' % self.region.endpoint, '<status>ok</status>', content_type='text/xml')
    conn = self.region.connect(aws_access_key_id='access_key', aws_secret_access_key='secret')
    resp = conn.get_status('getStatus', {'par1': 'foo', 'par2': 'baz'}, 'status')
    self.assertEqual(resp, 'ok')