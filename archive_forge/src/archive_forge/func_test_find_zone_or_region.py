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
def test_find_zone_or_region(self):
    zone1 = self.driver._find_zone_or_region('libcloud-demo-np-node', 'instances')
    self.assertEqual(zone1.name, 'us-central2-a')
    zone2 = self.driver._find_zone_or_region('libcloud-demo-europe-np-node', 'instances')
    self.assertEqual(zone2.name, 'europe-west1-a')
    region = self.driver._find_zone_or_region('libcloud-demo-address', 'addresses', region=True)
    self.assertEqual(region.name, 'us-central1')