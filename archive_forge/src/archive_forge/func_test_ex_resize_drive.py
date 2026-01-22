import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_resize_drive(self):
    drive = self.driver.ex_list_user_drives()[0]
    size = 10
    resized_drive = self.driver.ex_resize_drive(drive=drive, size=size)
    self.assertEqual(resized_drive.name, 'test drive 5')
    self.assertEqual(resized_drive.media, 'disk')
    self.assertEqual(resized_drive.size, size)