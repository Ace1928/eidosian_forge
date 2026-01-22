import os
import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlparse, parse_qsl, assertRaisesRegex
from libcloud.common.types import ProviderError
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import (
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudstack import CloudStackNodeDriver, CloudStackAffinityGroupType
def test_create_node_default_location_success(self):
    size = self.driver.list_sizes()[0]
    image = self.driver.list_images()[0]
    default_location = self.driver.list_locations()[0]
    node = self.driver.create_node(name='fred', image=image, size=size)
    self.assertEqual(node.name, 'fred')
    self.assertEqual(node.public_ips, [])
    self.assertEqual(node.private_ips, ['192.168.1.2'])
    self.assertEqual(node.extra['zone_id'], default_location.id)