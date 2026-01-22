import os
import mock
import boto
from boto.pyami.config import Config
from boto.regioninfo import RegionInfo, load_endpoint_json, merge_endpoints
from boto.regioninfo import load_regions, get_regions, connect
from tests.unit import unittest
def test_does_not_use_heuristics_by_default(self):
    connection = connect('ec2', 'us-southeast-43', connection_cls=FakeConn)
    self.assertIsNone(connection)