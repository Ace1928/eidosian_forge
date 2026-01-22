import os
import sys
import base64
from datetime import datetime
from collections import OrderedDict
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import b, httplib, parse_qs
from libcloud.compute.base import (
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import EC2_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ec2 import (
def test_attach(self):
    vol = StorageVolume(id='vol-4282672b', name='test', size=10, state=StorageVolumeState.AVAILABLE, driver=self.driver)
    node = Node('i-4382922a', None, None, None, None, self.driver)
    retValue = self.driver.attach_volume(node, vol, '/dev/sdh')
    self.assertTrue(retValue)