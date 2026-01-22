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
def test_get_object_by_kind(self):
    obj = self.driver._get_object_by_kind(None)
    self.assertIsNone(obj)
    obj = self.driver._get_object_by_kind('')
    self.assertIsNone(obj)
    obj = self.driver._get_object_by_kind('https://www.googleapis.com/compute/v1/projects/project_name/global/targetHttpProxies/web-proxy')
    self.assertEqual(obj.name, 'web-proxy')