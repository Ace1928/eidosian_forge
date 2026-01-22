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
def test_ex_list_backendservices(self):
    self.backendservices_mock = 'empty'
    backendservices_list = self.driver.ex_list_backendservices()
    self.assertListEqual(backendservices_list, [])
    self.backendservices_mock = 'web-service'
    backendservices_list = self.driver.ex_list_backendservices()
    web_service = backendservices_list[0]
    self.assertEqual(web_service.name, 'web-service')
    self.assertEqual(len(web_service.healthchecks), 1)
    self.assertEqual(len(web_service.backends), 2)