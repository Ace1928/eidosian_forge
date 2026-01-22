import sys
import base64
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import BRIGHTBOX_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.brightbox import BrightboxNodeDriver
def test_create_node_in_location(self):
    size = self.driver.list_sizes()[0]
    image = self.driver.list_images()[0]
    location = self.driver.list_locations()[1]
    node = self.driver.create_node(name='Test Node', image=image, size=size, location=location)
    self.assertEqual('srv-nnumd', node.id)
    self.assertEqual('Test Node', node.name)
    self.assertEqual('gb1-b', node.extra['zone'].name)