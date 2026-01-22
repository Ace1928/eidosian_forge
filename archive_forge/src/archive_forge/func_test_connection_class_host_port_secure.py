import sys
import unittest
from libcloud.common.types import LibcloudError
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.drivers.minio import MinIOStorageDriver, MinIOConnectionAWS4
def test_connection_class_host_port_secure(self):
    host = '127.0.0.3'
    port = 9000
    driver = self.driver_type(*self.driver_args, host=host, port=port, secure=False)
    self.assertEqual(driver.connectionCls.host, host)
    self.assertEqual(driver.connection.port, 9000)
    self.assertEqual(driver.connection.secure, False)
    host = '127.0.0.4'
    port = 9000
    driver = self.driver_type(*self.driver_args, host=host, port=port, secure=True)
    self.assertEqual(driver.connectionCls.host, host)
    self.assertEqual(driver.connection.port, 9000)
    self.assertEqual(driver.connection.secure, True)