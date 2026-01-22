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
def test_ex_list_firewall_rules_icmp(self):
    CloudStackMockHttp.fixture_tag = 'firewallicmp'
    rules = self.driver.ex_list_firewall_rules()
    self.assertEqual(len(rules), 1)
    rule = rules[0]
    self.assertEqual(rule.address.address, '1.1.1.116')
    self.assertEqual(rule.protocol, 'icmp')
    self.assertEqual(rule.cidr_list, '192.168.0.0/16')
    self.assertEqual(rule.icmp_code, 0)
    self.assertEqual(rule.icmp_type, 8)
    self.assertIsNone(rule.start_port)
    self.assertIsNone(rule.end_port)