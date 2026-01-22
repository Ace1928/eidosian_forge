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
def test_ex_list_availability_zones(self):
    availability_zones = self.driver.ex_list_availability_zones()
    availability_zone = availability_zones[0]
    self.assertTrue(len(availability_zones) > 0)
    self.assertEqual(availability_zone.name, 'eu-west-1a')
    self.assertEqual(availability_zone.zone_state, 'available')
    self.assertEqual(availability_zone.region_name, 'eu-west-1')