import os
import unittest
from boto.s3.keyfile import KeyFile
from tests.integration.s3.mock_storage_service import MockConnection
from tests.integration.s3.mock_storage_service import MockBucket
def testReadFull(self):
    self.assertEqual(self.keyfile.read(len(self.contents)), self.contents)