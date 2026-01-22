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
def test_create_node_with_labels(self):
    node_name = 'node-name'
    image = self.driver.ex_get_image('debian-7')
    size = self.driver.ex_get_size('n1-standard-1')
    zone = self.driver.ex_get_zone('us-central1-a')
    labels = {'label1': 'v1', 'label2': 'v2'}
    request, data = self.driver._create_node_req(node_name, size, image, zone, ex_labels=labels)
    self.assertTrue(data['labels'] is not None)
    self.assertEqual(len(data['labels']), 2)
    self.assertEqual(data['labels']['label1'], 'v1')
    self.assertEqual(data['labels']['label2'], 'v2')