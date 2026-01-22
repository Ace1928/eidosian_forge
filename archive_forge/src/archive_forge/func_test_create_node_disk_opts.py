import sys
import datetime
import unittest
from unittest import mock
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gce import (
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def test_create_node_disk_opts(self):
    node_name = 'node-name'
    size = self.driver.ex_get_size('n1-standard-1')
    image = self.driver.ex_get_image('debian-7')
    boot_disk = self.driver.ex_get_volume('lcdisk')
    disk_type = self.driver.ex_get_disktype('pd-ssd', 'us-central1-a')
    DEMO_BASE_NAME = 'lc-test'
    gce_disk_struct = [{'type': 'PERSISTENT', 'deviceName': '%s-gstruct' % DEMO_BASE_NAME, 'initializeParams': {'diskName': '%s-gstruct' % DEMO_BASE_NAME, 'sourceImage': image.extra['selfLink']}, 'boot': True, 'autoDelete': True}, {'type': 'SCRATCH', 'deviceName': '%s-gstruct-lssd' % DEMO_BASE_NAME, 'initializeParams': {'diskType': disk_type.extra['selfLink']}, 'autoDelete': True}]
    self.assertRaises(ValueError, self.driver.create_node, node_name, size, None)
    node = self.driver.create_node(node_name, size, image)
    self.assertTrue(isinstance(node, Node))
    node = self.driver.create_node(node_name, size, None, ex_boot_disk=boot_disk)
    self.assertTrue(isinstance(node, Node))
    node = self.driver.create_node(node_name, size, None, ex_disks_gce_struct=gce_disk_struct)
    self.assertTrue(isinstance(node, Node))
    self.assertRaises(ValueError, self.driver.create_node, node_name, size, None, ex_boot_disk=boot_disk, ex_disks_gce_struct=gce_disk_struct)