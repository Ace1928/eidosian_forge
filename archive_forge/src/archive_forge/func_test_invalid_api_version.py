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
def test_invalid_api_version(self):
    kwargs = {'api_version': '2.0'}
    self.driver = BrightboxNodeDriver(*BRIGHTBOX_PARAMS, **kwargs)
    self.assertRaises(Exception, self.driver.list_locations)