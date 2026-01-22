import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def test_list_key_pairs_UNAUTHORIZED(self):
    VultrMockHttpV2.type = 'UNAUTHORIZED'
    with self.assertRaises(VultrException):
        self.driver.list_key_pairs()