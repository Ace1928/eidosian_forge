import os
import mock
import boto
from boto.pyami.config import Config
from boto.regioninfo import RegionInfo, load_endpoint_json, merge_endpoints
from boto.regioninfo import load_regions, get_regions, connect
from tests.unit import unittest
def test_load_endpoint_json(self):
    endpoints = load_endpoint_json(boto.ENDPOINTS_PATH)
    self.assertTrue('partitions' in endpoints)