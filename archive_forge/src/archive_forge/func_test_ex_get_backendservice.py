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
def test_ex_get_backendservice(self):
    web_service = self.driver.ex_get_backendservice('web-service')
    self.assertEqual(web_service.name, 'web-service')
    self.assertEqual(web_service.protocol, 'HTTP')
    self.assertEqual(web_service.port, 80)
    self.assertEqual(web_service.timeout, 30)
    self.assertEqual(web_service.healthchecks[0].name, 'basic-check')
    self.assertEqual(len(web_service.healthchecks), 1)
    backends = web_service.backends
    self.assertEqual(len(backends), 2)
    self.assertEqual(backends[0]['balancingMode'], 'RATE')
    self.assertEqual(backends[0]['maxRate'], 100)
    self.assertEqual(backends[0]['capacityScaler'], 1.0)
    web_service = self.driver.ex_get_backendservice('no-backends')
    self.assertEqual(web_service.name, 'web-service')
    self.assertEqual(web_service.healthchecks[0].name, 'basic-check')
    self.assertEqual(len(web_service.healthchecks), 1)
    self.assertEqual(len(web_service.backends), 0)