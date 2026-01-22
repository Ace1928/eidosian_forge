import sys
import unittest
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.drivers.s3 import S3SignatureV4Connection
from libcloud.test.storage.test_s3 import S3Tests, S3MockHttp
from libcloud.storage.drivers.scaleway import SCW_FR_PAR_STANDARD_HOST, ScalewayStorageDriver
def test_connection_class_type(self):
    self.assertEqual(self.driver.connectionCls, S3SignatureV4Connection)