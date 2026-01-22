import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts
from libcloud.compute.base import Node
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import CloudSigmaNodeDriver, CloudSigmaZrhNodeDriver
def test_ex_set_node_configuration(self):
    node = self.driver.list_nodes()[0]
    result = self.driver.ex_set_node_configuration(node, **{'smp': 2})
    self.assertTrue(result)