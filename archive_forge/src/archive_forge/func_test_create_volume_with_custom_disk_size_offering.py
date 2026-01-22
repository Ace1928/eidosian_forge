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
def test_create_volume_with_custom_disk_size_offering(self):
    CloudStackMockHttp.fixture_tag = 'withcustomdisksize'
    volumeName = 'vol-0'
    location = self.driver.list_locations()[0]
    volume = self.driver.create_volume(10, volumeName, location)
    self.assertEqual(volumeName, volume.name)