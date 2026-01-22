import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume, NodeAuthSSHKey, NodeAuthPassword
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.linode import LinodeNodeDriver
def test_create_node_ssh_key_auth(self):
    node = self.driver.create_node(name='Test', location=self.driver.list_locations()[0], size=self.driver.list_sizes()[0], image=self.driver.list_images()[6], auth=NodeAuthSSHKey('foo'))
    self.assertTrue(isinstance(node, Node))