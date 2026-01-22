import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def test_ex_list_available_sizes_for_location(self):
    location = self.driver.list_locations()[0]
    available_sizes = self.driver.ex_list_available_sizes_for_location(location)
    self.assertTrue(isinstance(available_sizes, list))