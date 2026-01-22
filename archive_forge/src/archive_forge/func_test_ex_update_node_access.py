import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib, xmlrpclib
from libcloud.test.secrets import VCL_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vcl import VCLNodeDriver as VCL
def test_ex_update_node_access(self):
    node = self.driver.list_nodes(ipaddr='192.168.1.1')[0]
    node = self.driver.ex_update_node_access(node, ipaddr='192.168.1.2')
    self.assertEqual(node.name, 'CentOS 5.4 Base (32 bit VM)')
    self.assertEqual(node.state, NodeState.RUNNING)
    self.assertEqual(node.extra['pass'], 'ehkNGW')