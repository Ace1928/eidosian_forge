import os
import unittest
from boto.s3.keyfile import KeyFile
from tests.integration.s3.mock_storage_service import MockConnection
from tests.integration.s3.mock_storage_service import MockBucket
def testSeekCur(self):
    self.assertEqual(self.keyfile.read(1), self.contents[0])
    self.keyfile.seek(1, os.SEEK_CUR)
    self.assertEqual(self.keyfile.tell(), 2)
    self.assertEqual(self.keyfile.read(4), self.contents[2:6])