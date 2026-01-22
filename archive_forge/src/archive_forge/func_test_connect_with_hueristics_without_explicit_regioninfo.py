import os
import mock
import boto
from boto.pyami.config import Config
from boto.regioninfo import RegionInfo, load_endpoint_json, merge_endpoints
from boto.regioninfo import load_regions, get_regions, connect
from tests.unit import unittest
def test_connect_with_hueristics_without_explicit_regioninfo(self):
    os.environ['BOTO_USE_ENDPOINT_HEURISTICS'] = 'True'
    self.addCleanup(os.environ.pop, 'BOTO_USE_ENDPOINT_HEURISTICS')
    connection = connect('ec2', 'us-southeast-43', connection_cls=FakeConn)
    self.assertIsNotNone(connection)
    self.assertIsInstance(connection.region, RegionInfo)
    self.assertEqual(connection.region.name, 'us-southeast-43')
    expected_endpoint = 'ec2.us-southeast-43.amazonaws.com'
    self.assertEqual(connection.region.endpoint, expected_endpoint)