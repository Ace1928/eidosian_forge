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
def test_ex_allocate_address(self):
    elastic_ip = self.driver.ex_allocate_address()
    self.assertEqual('192.0.2.1', elastic_ip.ip)
    self.assertEqual('standard', elastic_ip.domain)
    EC2MockHttp.type = 'vpc'
    elastic_ip = self.driver.ex_allocate_address(domain='vpc')
    self.assertEqual('192.0.2.2', elastic_ip.ip)
    self.assertEqual('vpc', elastic_ip.domain)
    self.assertEqual('eipalloc-666d7f04', elastic_ip.extra['allocation_id'])