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
def test_create_node_req(self):
    image = self.driver.ex_get_image('debian-7')
    size = self.driver.ex_get_size('n1-standard-1')
    location = self.driver.zone
    network = self.driver.ex_get_network('default')
    tags = ['libcloud']
    metadata = [{'key': 'test_key', 'value': 'test_value'}]
    boot_disk = self.driver.ex_get_volume('lcdisk')
    node_request, node_data = self.driver._create_node_req('lcnode', size, image, location, network, tags, metadata, boot_disk)
    self.assertEqual(node_request, '/zones/%s/instances' % location.name)
    self.assertEqual(node_data['metadata']['items'][0]['key'], 'test_key')
    self.assertEqual(node_data['tags']['items'][0], 'libcloud')
    self.assertEqual(node_data['name'], 'lcnode')
    self.assertTrue(node_data['disks'][0]['boot'])
    self.assertIsInstance(node_data['serviceAccounts'], list)
    self.assertIsInstance(node_data['serviceAccounts'][0], dict)
    self.assertEqual(node_data['serviceAccounts'][0]['email'], 'default')
    self.assertIsInstance(node_data['serviceAccounts'][0]['scopes'], list)
    self.assertEqual(len(node_data['serviceAccounts'][0]['scopes']), 1)
    self.assertEqual(len(node_data['networkInterfaces']), 1)
    self.assertTrue(node_data['networkInterfaces'][0]['network'].startswith('https://'))