import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_start_node_already_started(self):
    CloudSigmaMockHttp.type = 'ALREADY_STARTED'
    expected_msg = 'Cannot start guest in state "started". Guest should be in state "stopped'
    assertRaisesRegex(self, CloudSigmaError, expected_msg, self.driver.ex_start_node, node=self.node)