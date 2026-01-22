import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_list_firewall_policies(self):
    policies = self.driver.ex_list_firewall_policies()
    policy = policies[1]
    rule = policy.rules[0]
    self.assertEqual(policy.name, 'My awesome policy')
    self.assertEqual(rule.action, 'drop')
    self.assertEqual(rule.direction, 'out')
    self.assertEqual(rule.dst_ip, '23.0.0.0/32')
    self.assertEqual(rule.ip_proto, 'tcp')
    self.assertIsNone(rule.dst_port)
    self.assertIsNone(rule.src_ip)
    self.assertIsNone(rule.src_port)
    self.assertEqual(rule.comment, 'Drop traffic from the VM to IP address 23.0.0.0/32')