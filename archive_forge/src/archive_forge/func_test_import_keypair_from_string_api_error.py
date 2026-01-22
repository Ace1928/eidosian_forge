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
def test_import_keypair_from_string_api_error(self):
    CloudStackMockHttp.type = 'api_error'
    name = 'test-pair'
    key_material = ''
    expected_msg = 'Public key is invalid'
    assertRaisesRegex(self, ProviderError, expected_msg, self.driver.import_key_pair_from_string, name=name, key_material=key_material)