import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_delete_tag(self):
    tag = self.driver.ex_list_tags()[0]
    status = self.driver.ex_delete_tag(tag=tag)
    self.assertTrue(status)