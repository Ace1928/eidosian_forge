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
def test_ex_create_port_forwarding_rule(self):
    node = self.driver.list_nodes()[0]
    address = self.driver.ex_list_public_ips()[0]
    private_port = 33
    private_end_port = 34
    public_port = 33
    public_end_port = 34
    openfirewall = True
    protocol = 'TCP'
    rule = self.driver.ex_create_port_forwarding_rule(node, address, private_port, public_port, protocol, public_end_port, private_end_port, openfirewall)
    self.assertEqual(rule.address, address)
    self.assertEqual(rule.protocol, protocol)
    self.assertEqual(rule.public_port, public_port)
    self.assertEqual(rule.public_end_port, public_end_port)
    self.assertEqual(rule.private_port, private_port)
    self.assertEqual(rule.private_end_port, private_end_port)