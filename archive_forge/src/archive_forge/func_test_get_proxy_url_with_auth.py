import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
def test_get_proxy_url_with_auth(self):
    conn = AWSAuthConnection('mockservice.cc-zone-1.amazonaws.com', aws_access_key_id='access_key', aws_secret_access_key='secret', suppress_consec_slashes=False, proxy='127.0.0.1', proxy_user='john.doe', proxy_pass='p4ssw0rd', proxy_port='8180')
    self.assertEqual(conn.get_proxy_url_with_auth(), 'http://john.doe:p4ssw0rd@127.0.0.1:8180')