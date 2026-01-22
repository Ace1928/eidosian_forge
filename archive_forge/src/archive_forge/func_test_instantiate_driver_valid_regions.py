import os
import sys
import base64
from datetime import datetime
from collections import OrderedDict
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import b, httplib, parse_qs
from libcloud.compute.base import (
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import EC2_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ec2 import (
def test_instantiate_driver_valid_regions(self):
    regions = VALID_EC2_REGIONS
    regions = [d for d in regions if d != 'nimbus' and d != 'cn-north-1']
    region_endpoints = [EC2NodeDriver(*EC2_PARAMS, **{'region': region}).connection.host for region in regions]
    self.assertEqual(len(region_endpoints), len(set(region_endpoints)), 'Multiple Region Drivers were given the same API endpoint')