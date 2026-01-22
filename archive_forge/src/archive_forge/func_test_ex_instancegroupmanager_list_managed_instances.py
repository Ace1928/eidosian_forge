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
def test_ex_instancegroupmanager_list_managed_instances(self):
    ig_name = 'myinstancegroup'
    ig_zone = 'us-central1-a'
    mig = self.driver.ex_get_instancegroupmanager(ig_name, ig_zone)
    instances = mig.list_managed_instances()
    self.assertTrue(all([x['currentAction'] == 'NONE' for x in instances]))
    self.assertTrue('base-foo-2vld' in [x['name'] for x in instances])
    self.assertEqual(len(instances), 4)