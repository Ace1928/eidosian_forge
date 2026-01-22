import os
import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlparse, parse_qsl, assertRaisesRegex
from libcloud.common.types import ProviderError
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import (
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudstack import CloudStackNodeDriver, CloudStackAffinityGroupType
def test_ex_list_affinity_groups(self):
    res = self.driver.ex_list_affinity_groups()
    self.assertEqual(len(res), 1)
    self.assertEqual(res[0].id, '11112')
    self.assertEqual(res[0].name, 'MyAG')
    self.assertIsInstance(res[0].type, CloudStackAffinityGroupType)
    self.assertEqual(res[0].type.type, 'MyAGType')