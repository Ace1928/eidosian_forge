import copy
import pickle
import os
from tests.compat import unittest, mock
from tests.unit import MockServiceWithConfigTestCase
from nose.tools import assert_equal
from boto.auth import HmacAuthV4Handler
from boto.auth import S3HmacAuthV4Handler
from boto.auth import detect_potential_s3sigv4
from boto.auth import detect_potential_sigv4
from boto.connection import HTTPRequest
from boto.provider import Provider
from boto.regioninfo import RegionInfo
def test_headers_to_sign(self):
    auth = HmacAuthV4Handler('glacier.us-east-1.amazonaws.com', mock.Mock(), self.provider)
    request = HTTPRequest('GET', 'http', 'glacier.us-east-1.amazonaws.com', 80, 'x/./././x .html', None, {}, {'x-amz-glacier-version': '2012-06-01'}, '')
    headers = auth.headers_to_sign(request)
    self.assertEqual(headers['Host'], 'glacier.us-east-1.amazonaws.com')
    request = HTTPRequest('GET', 'https', 'glacier.us-east-1.amazonaws.com', 443, 'x/./././x .html', None, {}, {'x-amz-glacier-version': '2012-06-01'}, '')
    headers = auth.headers_to_sign(request)
    self.assertEqual(headers['Host'], 'glacier.us-east-1.amazonaws.com')
    request = HTTPRequest('GET', 'https', 'glacier.us-east-1.amazonaws.com', 8080, 'x/./././x .html', None, {}, {'x-amz-glacier-version': '2012-06-01'}, '')
    headers = auth.headers_to_sign(request)
    self.assertEqual(headers['Host'], 'glacier.us-east-1.amazonaws.com:8080')