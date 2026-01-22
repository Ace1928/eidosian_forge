import sys
import unittest
from libcloud.test import LibcloudTestCase
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.drivers.digitalocean_spaces import (
def test_invalid_region(self):
    with self.assertRaises(LibcloudError):
        self.driver_type(*self.driver_args, region='atlantis', host=self.default_host)