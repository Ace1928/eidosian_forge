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
def test_determine_service_name(self):
    name = self.auth.determine_service_name('s3.us-west-2.amazonaws.com')
    self.assertEqual(name, 's3')
    name = self.auth.determine_service_name('s3-us-west-2.amazonaws.com')
    self.assertEqual(name, 's3')
    name = self.auth.determine_service_name('bucket.s3.us-west-2.amazonaws.com')
    self.assertEqual(name, 's3')
    name = self.auth.determine_service_name('bucket.s3-us-west-2.amazonaws.com')
    self.assertEqual(name, 's3')