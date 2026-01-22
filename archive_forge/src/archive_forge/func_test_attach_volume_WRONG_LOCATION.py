import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def test_attach_volume_WRONG_LOCATION(self):
    volume = self.driver.list_volumes()[1]
    node = self.driver.list_nodes()[0]
    VultrMockHttpV2.type = 'WRONG_LOCATION'
    with self.assertRaises(VultrException):
        self.driver.attach_volume(node, volume)