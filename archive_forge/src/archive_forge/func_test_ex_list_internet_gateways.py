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
def test_ex_list_internet_gateways(self):
    gateways = self.driver.ex_list_internet_gateways()
    self.assertEqual(len(gateways), 2)
    self.assertEqual('igw-84dd3ae1', gateways[0].id)
    self.assertEqual('igw-7fdae215', gateways[1].id)
    self.assertEqual('available', gateways[1].state)
    self.assertEqual('vpc-62cad41e', gateways[1].vpc_id)