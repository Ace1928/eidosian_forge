import os
import unittest
from boto.s3.keyfile import KeyFile
from tests.integration.s3.mock_storage_service import MockConnection
from tests.integration.s3.mock_storage_service import MockBucket
def testTell(self):
    self.assertEqual(self.keyfile.tell(), 0)
    self.keyfile.read(4)
    self.assertEqual(self.keyfile.tell(), 4)
    self.keyfile.read(6)
    self.assertEqual(self.keyfile.tell(), 10)
    self.keyfile.close()
    try:
        self.keyfile.tell()
    except ValueError as e:
        self.assertEqual(str(e), 'I/O operation on closed file')