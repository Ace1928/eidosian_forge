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
def test_create_volume_with_defined_volume_type(self):
    CloudStackMockHttp.fixture_tag = 'withvolumetype'
    volumeName = 'vol-0'
    volLocation = self.driver.list_locations()[0]
    diskOffering = self.driver.ex_list_disk_offerings()[0]
    volumeType = diskOffering.name
    volume = self.driver.create_volume(10, volumeName, location=volLocation, ex_volume_type=volumeType)
    self.assertEqual(volumeName, volume.name)