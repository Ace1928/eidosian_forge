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
def test_ex_get_size(self):
    size_name = 'n1-standard-1'
    size = self.driver.ex_get_size(size_name)
    self.assertEqual(size.name, size_name)
    self.assertEqual(size.extra['zone'].name, 'us-central1-a')
    self.assertEqual(size.disk, 10)
    self.assertEqual(size.ram, 3840)
    self.assertEqual(size.extra['guestCpus'], 1)