import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_create_drive(self):
    CloudSigmaMockHttp.type = 'CREATE'
    name = 'test drive 5'
    size = 2000 * 1024 * 1024
    drive = self.driver.ex_create_drive(name=name, size=size, media='disk')
    self.assertEqual(drive.name, 'test drive 5')
    self.assertEqual(drive.media, 'disk')