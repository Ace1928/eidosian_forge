import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import ServiceUnavailableError
from libcloud.compute.base import NodeSize, NodeImage
from libcloud.test.secrets import VULTR_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV1
def test_delete_key_pair_success(self):
    key_pairs = self.driver.list_key_pairs()
    key_pair = key_pairs[0]
    res = self.driver.delete_key_pair(key_pair)
    self.assertTrue(res)