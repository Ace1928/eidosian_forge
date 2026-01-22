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
def test_non_string_headers(self):
    self.awesome_bucket_request.headers['Content-Length'] = 8
    self.awesome_bucket_request.headers['x-amz-server-side-encryption-customer-key-md5'] = 2
    self.awesome_bucket_request.headers['x-amz-server-side-encryption-customer-key'] = 1
    canonical_headers = self.auth.canonical_headers(self.awesome_bucket_request.headers)
    self.assertEqual(canonical_headers, 'content-length:8\nuser-agent:Boto\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20130605T193245Z\nx-amz-server-side-encryption-customer-key:1\nx-amz-server-side-encryption-customer-key-md5:2')