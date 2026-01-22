import sys
import unittest
from libcloud.test import LibcloudTestCase
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.drivers.digitalocean_spaces import (
def test_connection_class_host(self):
    host = self.driver_v2.connectionCls.host
    self.assertEqual(host, self.default_host)
    host = self.driver_v4.connectionCls.host
    self.assertEqual(host, self.alt_host)