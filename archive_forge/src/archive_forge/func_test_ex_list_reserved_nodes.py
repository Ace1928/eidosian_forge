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
def test_ex_list_reserved_nodes(self):
    node = self.driver.ex_list_reserved_nodes()[0]
    self.assertEqual(node.id, '93bbbca2-c500-49d0-9ede-9d8737400498')
    self.assertEqual(node.state, 'active')
    self.assertEqual(node.extra['instance_type'], 't1.micro')
    self.assertEqual(node.extra['availability'], 'us-east-1b')
    self.assertEqual(node.extra['start'], '2013-06-18T12:07:53.161Z')
    self.assertEqual(node.extra['end'], '2014-06-18T12:07:53.161Z')
    self.assertEqual(node.extra['duration'], 31536000)
    self.assertEqual(node.extra['usage_price'], 0.012)
    self.assertEqual(node.extra['fixed_price'], 23.0)
    self.assertEqual(node.extra['instance_count'], 1)
    self.assertEqual(node.extra['description'], 'Linux/UNIX')
    self.assertEqual(node.extra['instance_tenancy'], 'default')
    self.assertEqual(node.extra['currency_code'], 'USD')
    self.assertEqual(node.extra['offering_type'], 'Light Utilization')