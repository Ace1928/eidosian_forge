import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import DIGITALOCEAN_v2_PARAMS
from libcloud.test.file_fixtures import FileFixtures
from libcloud.common.digitalocean import DigitalOceanBaseDriver
def test_ex_account_info(self):
    account_info = self.driver.ex_account_info()
    self.assertEqual(account_info['uuid'], 'a1234567890b1234567890c1234567890d12345')
    self.assertTrue(account_info['email_verified'])
    self.assertEqual(account_info['email'], 'user@domain.tld')
    self.assertEqual(account_info['droplet_limit'], 10)