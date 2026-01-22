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
def test_inner_whitespace_is_collapsed(self):
    auth = HmacAuthV4Handler('glacier.us-east-1.amazonaws.com', mock.Mock(), self.provider)
    self.request.headers['x-amz-archive-description'] = 'two  spaces'
    self.request.headers['x-amz-quoted-string'] = '  "a   b   c" '
    headers = auth.headers_to_sign(self.request)
    self.assertEqual(headers, {'Host': 'glacier.us-east-1.amazonaws.com', 'x-amz-archive-description': 'two  spaces', 'x-amz-glacier-version': '2012-06-01', 'x-amz-quoted-string': '  "a   b   c" '})
    self.assertEqual(auth.canonical_headers(headers), 'host:glacier.us-east-1.amazonaws.com\nx-amz-archive-description:two spaces\nx-amz-glacier-version:2012-06-01\nx-amz-quoted-string:"a   b   c"')