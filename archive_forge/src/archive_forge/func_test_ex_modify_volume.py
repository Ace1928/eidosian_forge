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
def test_ex_modify_volume(self):
    volume = self.driver.list_volumes()[0]
    assert volume.id == 'vol-10ae5e2b'
    params = {'VolumeType': 'io1', 'Size': 2, 'Iops': 1000}
    volume_modification = self.driver.ex_modify_volume(volume, params)
    self.assertIsNone(volume_modification.end_time)
    self.assertEqual('modifying', volume_modification.modification_state)
    self.assertEqual(300, volume_modification.original_iops)
    self.assertEqual(1, volume_modification.original_size)
    self.assertEqual('gp2', volume_modification.original_volume_type)
    self.assertEqual(0, volume_modification.progress)
    self.assertIsNone(volume_modification.status_message)
    self.assertEqual(1000, volume_modification.target_iops)
    self.assertEqual(2, volume_modification.target_size)
    self.assertEqual('io1', volume_modification.target_volume_type)
    self.assertEqual('vol-10ae5e2b', volume_modification.volume_id)