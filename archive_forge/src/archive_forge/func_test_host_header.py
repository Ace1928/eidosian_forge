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
def test_host_header(self):
    host = self.auth.host_header(self.awesome_bucket_request.host, self.awesome_bucket_request)
    self.assertEqual(host, 'awesome-bucket.s3-us-west-2.amazonaws.com')