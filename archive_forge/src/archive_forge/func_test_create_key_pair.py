import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import NodeImage
from libcloud.test.secrets import DIGITALOCEAN_v1_PARAMS, DIGITALOCEAN_v2_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.digitalocean import DigitalOcean_v1_Error
from libcloud.compute.drivers.digitalocean import DigitalOceanNodeDriver
def test_create_key_pair(self):
    DigitalOceanMockHttp.type = 'CREATE'
    key = self.driver.create_key_pair(name='test1', public_key='ssh-rsa AAAAB3NzaC1yc2EAAAADAQsxRiUKn example')
    self.assertEqual(key.name, 'test1')
    self.assertEqual(key.fingerprint, 'f5:d1:78:ed:28:72:5f:e1:ac:94:fd:1f:e0:a3:48:6d')