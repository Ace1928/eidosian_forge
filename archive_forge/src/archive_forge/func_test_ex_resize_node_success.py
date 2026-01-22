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
def test_ex_resize_node_success(self):
    node = self.driver.list_nodes()[0]
    size = self.driver.list_sizes()[0]
    DigitalOceanMockHttp.type = 'RESIZE'
    result = self.driver.ex_resize_node(node, size)
    self.assertTrue(result)