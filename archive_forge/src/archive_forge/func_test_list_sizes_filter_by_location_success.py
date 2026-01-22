import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import NodeImage
from libcloud.test.secrets import DIGITALOCEAN_v1_PARAMS, DIGITALOCEAN_v2_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.digitalocean import DigitalOcean_v1_Error
from libcloud.compute.drivers.digitalocean import DigitalOceanNodeDriver
def test_list_sizes_filter_by_location_success(self):
    location = self.driver.list_locations()[1]
    sizes = self.driver.list_sizes(location=location)
    self.assertTrue(len(sizes) >= 1)
    size = sizes[0]
    self.assertTrue(size.id is not None)
    self.assertEqual(size.name, '512mb')
    self.assertTrue(location.id in size.extra['regions'])
    location = self.driver.list_locations()[1]
    location.id = 'doesntexist'
    sizes = self.driver.list_sizes(location=location)
    self.assertEqual(len(sizes), 0)