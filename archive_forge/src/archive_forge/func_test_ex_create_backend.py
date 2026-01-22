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
def test_ex_create_backend(self):
    ig = self.driver.ex_get_instancegroup('myinstancegroup', 'us-central1-a')
    backend = self.driver.ex_create_backend(ig)
    self.assertTrue(isinstance(backend, GCEBackend))
    self.assertEqual(backend.name, '{}/instanceGroups/{}'.format(ig.zone.name, ig.name))
    self.assertEqual(backend.instance_group.name, ig.name)
    self.assertEqual(backend.balancing_mode, 'UTILIZATION')