import os
import mock
import boto
from boto.pyami.config import Config
from boto.regioninfo import RegionInfo, load_endpoint_json, merge_endpoints
from boto.regioninfo import load_regions, get_regions, connect
from tests.unit import unittest
def test_uses_region_override(self):
    connection = connect('ec2', 'us-west-2', connection_cls=FakeConn, region_cls=TestRegionInfo)
    self.assertIsInstance(connection.region, TestRegionInfo)
    self.assertEqual(connection.region.name, 'us-west-2')
    expected_endpoint = 'ec2.us-west-2.amazonaws.com'
    self.assertEqual(connection.region.endpoint, expected_endpoint)