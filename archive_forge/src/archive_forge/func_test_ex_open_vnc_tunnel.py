import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_open_vnc_tunnel(self):
    node = self.driver.list_nodes()[0]
    vnc_url = self.driver.ex_open_vnc_tunnel(node=node)
    self.assertEqual(vnc_url, 'vnc://direct.lvs.cloudsigma.com:41111')