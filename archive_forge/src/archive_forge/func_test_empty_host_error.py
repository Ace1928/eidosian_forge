import sys
import unittest
from libcloud.common.types import LibcloudError
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.drivers.minio import MinIOStorageDriver, MinIOConnectionAWS4
def test_empty_host_error(self):
    self.assertRaisesRegex(LibcloudError, 'host argument is required', self.driver_type, *self.driver_args)