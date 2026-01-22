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
def test_add_auth(self):
    self.assertFalse('x-amz-content-sha256' in self.request.headers)
    self.auth.add_auth(self.request)
    self.assertTrue('x-amz-content-sha256' in self.request.headers)
    the_sha = self.request.headers['x-amz-content-sha256']
    self.assertEqual(the_sha, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')