import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import ServiceUnavailableError
from libcloud.compute.base import NodeSize, NodeImage
from libcloud.test.secrets import VULTR_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV1
def test_list_key_pairs_success(self):
    key_pairs = self.driver.list_key_pairs()
    self.assertEqual(len(key_pairs), 1)
    key_pair = key_pairs[0]
    self.assertEqual(key_pair.id, '5806a8ef2a0c6')
    self.assertEqual(key_pair.name, 'test-key-pair')