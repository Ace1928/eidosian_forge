import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.compute.base import NodeImage
from libcloud.test.secrets import SCALEWAY_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.scaleway import ScalewayNodeDriver
def test_create_node_success(self):
    image = self.driver.list_images()[0]
    size = self.driver.list_sizes()[0]
    location = self.driver.list_locations()[0]
    ScalewayMockHttp.type = 'POST'
    node = self.driver.create_node(name='test', size=size, image=image, region=location)
    self.assertEqual(node.name, 'my_server')
    self.assertEqual(node.public_ips, [])
    self.assertEqual(node.extra['volumes']['0']['id'], 'd9257116-6919-49b4-a420-dcfdff51fcb1')
    self.assertEqual(node.extra['organization'], '000a115d-2852-4b0a-9ce8-47f1134ba95a')