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
def test_ex_create_firewall_rule_icmp(self):
    address = self.driver.ex_list_public_ips()[0]
    cidr_list = '192.168.0.0/16'
    protocol = 'icmp'
    icmp_code = 0
    icmp_type = 8
    rule = self.driver.ex_create_firewall_rule(address, cidr_list, protocol, icmp_code=icmp_code, icmp_type=icmp_type)
    self.assertEqual(rule.address, address)
    self.assertEqual(rule.protocol, protocol)
    self.assertEqual(rule.icmp_code, 0)
    self.assertEqual(rule.icmp_type, 8)
    self.assertIsNone(rule.start_port)
    self.assertIsNone(rule.end_port)