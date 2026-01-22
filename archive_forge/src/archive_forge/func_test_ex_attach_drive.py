import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_attach_drive(self):
    node_id = '3df825cb-9c1b-470d-acbd-03e1a966c046'
    node = self.driver.ex_get_node(node_id)
    drives = self.driver.ex_list_user_drives()
    drive = drives[0]
    response = self.driver.ex_attach_drive(node, drive)
    self.assertTrue(response)