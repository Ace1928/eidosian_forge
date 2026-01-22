import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_toggle_subscription_auto_renew(self):
    subscription = self.driver.ex_list_subscriptions()[0]
    status = self.driver.ex_toggle_subscription_auto_renew(subscription=subscription)
    self.assertTrue(status)