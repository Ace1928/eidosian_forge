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
def test_import_keypair_from_file(self):
    fingerprint = 'c4:a1:e5:d4:50:84:a9:4c:6b:22:ee:d6:57:02:b8:15'
    path = os.path.join(os.path.dirname(__file__), 'fixtures', 'cloudstack', 'dummy_rsa.pub')
    key_pair = self.driver.import_key_pair_from_file('foobar', path)
    self.assertEqual(key_pair.name, 'foobar')
    self.assertEqual(key_pair.fingerprint, fingerprint)
    res = self.driver.ex_import_keypair('foobar', path)
    self.assertEqual(res['keyName'], 'foobar')
    self.assertEqual(res['keyFingerprint'], fingerprint)