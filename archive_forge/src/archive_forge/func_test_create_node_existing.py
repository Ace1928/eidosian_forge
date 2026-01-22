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
def test_create_node_existing(self):
    node_name = 'libcloud-demo-europe-np-node'
    image = self.driver.ex_get_image('debian-7')
    size = self.driver.ex_get_size('n1-standard-1', zone='europe-west1-a')
    self.assertRaises(ResourceExistsError, self.driver.create_node, node_name, size, image, location='europe-west1-a')