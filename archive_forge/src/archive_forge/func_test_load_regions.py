import os
import mock
import boto
from boto.pyami.config import Config
from boto.regioninfo import RegionInfo, load_endpoint_json, merge_endpoints
from boto.regioninfo import load_regions, get_regions, connect
from tests.unit import unittest
def test_load_regions(self):
    endpoints = load_regions()
    self.assertTrue('us-east-1' in endpoints['ec2'])
    self.assertFalse('test-1' in endpoints['ec2'])
    os.environ['BOTO_ENDPOINTS'] = os.path.join(os.path.dirname(__file__), 'test_endpoints.json')
    self.addCleanup(os.environ.pop, 'BOTO_ENDPOINTS')
    endpoints = load_regions()
    self.assertTrue('us-east-1' in endpoints['ec2'])
    self.assertTrue('test-1' in endpoints['ec2'])
    self.assertEqual(endpoints['ec2']['test-1'], 'ec2.test-1.amazonaws.com')