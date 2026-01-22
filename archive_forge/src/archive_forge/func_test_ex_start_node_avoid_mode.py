import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_start_node_avoid_mode(self):
    CloudSigmaMockHttp.type = 'AVOID_MODE'
    ex_avoid = ['1', '2']
    status = self.driver.ex_start_node(node=self.node, ex_avoid=ex_avoid)
    self.assertTrue(status)