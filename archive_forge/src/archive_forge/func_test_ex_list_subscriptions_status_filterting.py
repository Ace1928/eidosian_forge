import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_list_subscriptions_status_filterting(self):
    CloudSigmaMockHttp.type = 'STATUS_FILTER'
    self.driver.ex_list_subscriptions(status='active')