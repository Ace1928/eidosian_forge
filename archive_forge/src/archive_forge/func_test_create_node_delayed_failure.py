import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib, urlparse, parse_qsl
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ktucloud import KTUCloudNodeDriver
def test_create_node_delayed_failure(self):
    size = self.driver.list_sizes()[0]
    image = self.driver.list_images()[0]
    KTUCloudStackMockHttp.fixture_tag = 'deployfail2'
    self.assertRaises(Exception, self.driver.create_node, name='node-name', image=image, size=size)