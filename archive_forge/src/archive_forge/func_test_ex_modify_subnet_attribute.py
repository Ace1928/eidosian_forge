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
def test_ex_modify_subnet_attribute(self):
    subnet = self.driver.ex_list_subnets()[0]
    resp = self.driver.ex_modify_subnet_attribute(subnet, 'auto_public_ip', True)
    self.assertTrue(resp)
    resp = self.driver.ex_modify_subnet_attribute(subnet, 'auto_ipv6', False)
    self.assertTrue(resp)
    expected_msg = 'Unsupported attribute: invalid'
    self.assertRaisesRegex(ValueError, expected_msg, self.driver.ex_modify_subnet_attribute, subnet, 'invalid', True)