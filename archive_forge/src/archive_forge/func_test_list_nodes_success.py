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
def test_list_nodes_success(self):
    nodes = self.driver.list_nodes()
    self.assertEqual(len(nodes), 2)
    self.assertEqual(nodes[0].name, 'my_server')
    self.assertEqual(nodes[0].public_ips, [])
    self.assertEqual(nodes[0].extra['volumes']['0']['id'], 'c1eb8f3a-4f0b-4b95-a71c-93223e457f5a')
    self.assertEqual(nodes[0].extra['organization'], '000a115d-2852-4b0a-9ce8-47f1134ba95a')