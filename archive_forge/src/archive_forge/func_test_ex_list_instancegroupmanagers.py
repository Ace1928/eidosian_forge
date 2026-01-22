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
def test_ex_list_instancegroupmanagers(self):
    instancegroupmanagers = self.driver.ex_list_instancegroupmanagers()
    instancegroupmanagers_all = self.driver.ex_list_instancegroupmanagers('all')
    instancegroupmanagers_ue1b = self.driver.ex_list_instancegroupmanagers('us-east1-b')
    self.assertEqual(len(instancegroupmanagers), 1)
    self.assertEqual(len(instancegroupmanagers_all), 2)
    self.assertEqual(len(instancegroupmanagers_ue1b), 1)