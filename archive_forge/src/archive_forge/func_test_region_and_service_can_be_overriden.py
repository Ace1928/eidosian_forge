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
def test_region_and_service_can_be_overriden(self):
    auth = HmacAuthV4Handler('queue.amazonaws.com', mock.Mock(), self.provider)
    self.request.headers['X-Amz-Date'] = '20121121000000'
    auth.region_name = 'us-west-2'
    auth.service_name = 'sqs'
    scope = auth.credential_scope(self.request)
    self.assertEqual(scope, '20121121/us-west-2/sqs/aws4_request')