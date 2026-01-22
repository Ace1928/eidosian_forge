import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_delete_firewall_policy(self):
    policy = self.driver.ex_list_firewall_policies()[0]
    status = self.driver.ex_delete_firewall_policy(policy=policy)
    self.assertTrue(status)