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
def test_get_region_from_zone(self):
    zone1 = self.driver.ex_get_zone('us-central1-a')
    expected_region1 = 'us-central1'
    region1 = self.driver._get_region_from_zone(zone1)
    self.assertEqual(region1.name, expected_region1)
    zone2 = self.driver.ex_get_zone('europe-west1-b')
    expected_region2 = 'europe-west1'
    region2 = self.driver._get_region_from_zone(zone2)
    self.assertEqual(region2.name, expected_region2)