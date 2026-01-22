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
def test_ex_create_instancegroupmanager(self):
    name = 'myinstancegroup'
    zone = 'us-central1-a'
    size = 4
    template_name = 'my-instance-template1'
    template = self.driver.ex_get_instancetemplate(template_name)
    mig = self.driver.ex_create_instancegroupmanager(name, zone, template, size, base_instance_name='base-foo')
    self.assertEqual(mig.name, name)
    self.assertEqual(mig.size, size)
    self.assertEqual(mig.zone.name, zone)