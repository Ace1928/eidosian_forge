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
def test_ex_set_common_instance_metadata(self):
    self.assertRaises(ValueError, self.driver.ex_set_common_instance_metadata, ['bad', 'type'])
    pydict = {'key': 'pydict', 'value': 1}
    self.driver.ex_set_common_instance_metadata(pydict)
    bad_gcedict = {'items': 'foo'}
    self.assertRaises(ValueError, self.driver.ex_set_common_instance_metadata, bad_gcedict)
    gcedict = {'items': [{'key': 'gcedict1', 'value': 'v1'}, {'key': 'gcedict2', 'value': 'v2'}]}
    self.driver.ex_set_common_instance_metadata(gcedict)
    project = GCEProject(id=None, name=None, metadata=None, quotas=None, driver=self.driver)
    project.set_common_instance_metadata(metadata=gcedict)