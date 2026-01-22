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
def test_create_node_location_not_provided_inferred_from_size(self):
    node_name = 'node-name'
    size = self.driver.ex_get_size('n1-standard-1')
    image = self.driver.ex_get_image('debian-7')
    zone = self.driver.ex_list_zones()[0]
    zone = self.driver.ex_get_zone('us-central1-a')
    self.driver.zone = None
    size.extra['zone'] = zone
    node = self.driver.create_node(node_name, size, image)
    self.assertTrue(node)