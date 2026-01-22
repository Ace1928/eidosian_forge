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
def test_node_ex_delete_port_forwarding_rule(self):
    node = self.driver.list_nodes()[0]
    self.assertEqual(len(node.extra['port_forwarding_rules']), 1)
    node.extra['port_forwarding_rules'][0].delete()
    self.assertEqual(len(node.extra['port_forwarding_rules']), 0)