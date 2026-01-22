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
def test_ex_create_urlmap(self):
    urlmap_name = 'web-map'
    for service in ('web-service', self.driver.ex_get_backendservice('web-service')):
        urlmap = self.driver.ex_create_urlmap(urlmap_name, service)
        self.assertTrue(isinstance(urlmap, GCEUrlMap))
        self.assertEqual(urlmap_name, urlmap.name)