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
def test_ex_create_floating_ip(self):
    nyc1 = [r for r in self.driver.list_locations() if r.id == 'nyc1'][0]
    floating_ip = self.driver.ex_create_floating_ip(nyc1)
    self.assertEqual(floating_ip.id, '167.138.123.111')
    self.assertEqual(floating_ip.ip_address, '167.138.123.111')
    self.assertEqual(floating_ip.extra['region']['slug'], 'nyc1')
    self.assertIsNone(floating_ip.node_id)