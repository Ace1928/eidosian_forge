import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import DIGITALOCEAN_v2_PARAMS
from libcloud.test.file_fixtures import FileFixtures
from libcloud.common.digitalocean import DigitalOceanBaseDriver
def test_authentication(self):
    DigitalOceanMockHttp.type = 'UNAUTHORIZED'
    self.assertRaises(InvalidCredsError, self.driver.ex_account_info)