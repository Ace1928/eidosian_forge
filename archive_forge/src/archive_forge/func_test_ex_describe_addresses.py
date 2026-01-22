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
def test_ex_describe_addresses(self):
    node = Node('i-4382922a', None, None, None, None, self.driver)
    nodes_elastic_ips = self.driver.ex_describe_addresses([node])
    self.assertEqual(len(nodes_elastic_ips), 1)
    self.assertEqual(len(nodes_elastic_ips[node.id]), 0)