import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeImage, NodeState, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.common.linode import LinodeDisk, LinodeIPAddress, LinodeExceptionV4
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.linode import LinodeNodeDriver, LinodeNodeDriverV4
def test_ex_list_node_addresses(self):
    node = Node('22344420', None, None, None, None, driver=self.driver)
    ips = self.driver.ex_list_node_addresses(node)
    ip = ips[0]
    self.assertEqual(ip.inet, '176.58.100.100')
    self.assertEqual(ip.version, 'ipv4')
    self.assertTrue(ip.public)
    for ip in ips:
        self.assertIsInstance(ip, LinodeIPAddress)
        self.assertEqual(node.id, str(ip.extra['node_id']))