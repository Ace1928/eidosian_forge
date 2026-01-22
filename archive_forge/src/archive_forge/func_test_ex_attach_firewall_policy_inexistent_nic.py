import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_attach_firewall_policy_inexistent_nic(self):
    policy = self.driver.ex_list_firewall_policies()[0]
    node = self.driver.list_nodes()[0]
    nic_mac = 'inexistent'
    expected_msg = 'Cannot find the NIC interface to attach a policy to'
    assertRaisesRegex(self, ValueError, expected_msg, self.driver.ex_attach_firewall_policy, policy=policy, node=node, nic_mac=nic_mac)