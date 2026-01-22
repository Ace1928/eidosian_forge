import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts
from libcloud.compute.base import Node
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import CloudSigmaNodeDriver, CloudSigmaZrhNodeDriver
def test_shutdown_node(self):
    nodes = self.driver.list_nodes()
    node = nodes[0]
    self.assertTrue(self.driver.ex_stop_node(node))
    self.assertTrue(self.driver.ex_shutdown_node(node))