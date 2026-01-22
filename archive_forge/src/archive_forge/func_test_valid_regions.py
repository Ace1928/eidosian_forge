import sys
import unittest
from libcloud.test import LibcloudTestCase
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.drivers.digitalocean_spaces import (
def test_valid_regions(self):
    for region, hostname in DO_SPACES_HOSTS_BY_REGION.items():
        driver = self.driver_type(*self.driver_args, region=region)
        self.assertEqual(driver.connectionCls.host, hostname)
        self.assertTrue(driver.connectionCls.host.startswith(region))