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
def test_ex_attach_floating_ip_to_node(self):
    node = self.driver.list_nodes()[0]
    floating_ip = self.driver.ex_get_floating_ip('133.166.122.204')
    ret = self.driver.ex_attach_floating_ip_to_node(node, floating_ip)
    self.assertTrue(ret)