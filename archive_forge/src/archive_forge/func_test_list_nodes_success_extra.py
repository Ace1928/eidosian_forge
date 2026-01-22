import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import ServiceUnavailableError
from libcloud.compute.base import NodeSize, NodeImage
from libcloud.test.secrets import VULTR_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV1
def test_list_nodes_success_extra(self):
    extra_keys = ['default_password', 'pending_charges', 'cost_per_month']
    nodes = self.driver.list_nodes()
    for node in nodes:
        self.assertTrue(len(node.extra.keys()) > 5)
        self.assertTrue(all((item in node.extra.keys() for item in extra_keys)))