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
def test_ex_describe_volumes_modifications(self):
    modifications = self.driver.ex_describe_volumes_modifications()
    self.assertEqual(len(modifications), 2)
    self.assertIsNone(modifications[0].end_time)
    self.assertEqual('optimizing', modifications[0].modification_state)
    self.assertEqual(100, modifications[0].original_iops)
    self.assertEqual(10, modifications[0].original_size)
    self.assertEqual('gp2', modifications[0].original_volume_type)
    self.assertEqual(3, modifications[0].progress)
    self.assertIsNone(modifications[0].status_message)
    self.assertEqual(10000, modifications[0].target_iops)
    self.assertEqual(2000, modifications[0].target_size)
    self.assertEqual('io1', modifications[0].target_volume_type)
    self.assertEqual('vol-06397e7a0eEXAMPLE', modifications[0].volume_id)
    self.assertEqual('completed', modifications[1].modification_state)
    self.assertEqual(100, modifications[1].original_iops)
    self.assertEqual(8, modifications[1].original_size)
    self.assertEqual('gp2', modifications[1].original_volume_type)
    self.assertEqual(100, modifications[1].progress)
    self.assertIsNone(modifications[1].status_message)
    self.assertEqual(10000, modifications[1].target_iops)
    self.assertEqual(200, modifications[1].target_size)
    self.assertEqual('io1', modifications[1].target_volume_type)
    self.assertEqual('vol-bEXAMPLE', modifications[1].volume_id)