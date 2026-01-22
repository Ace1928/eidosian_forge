import sys
import base64
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import BRIGHTBOX_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.brightbox import BrightboxNodeDriver
def test_other_host(self):
    kwargs = {'host': 'api.gbt.brightbox.com'}
    self.driver = BrightboxNodeDriver(*BRIGHTBOX_PARAMS, **kwargs)
    locations = self.driver.list_locations()
    self.assertEqual(len(locations), 0)