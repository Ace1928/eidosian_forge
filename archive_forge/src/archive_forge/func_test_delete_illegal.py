import unittest
import time
from boto.s3.key import Key
from boto.s3.deletemarker import DeleteMarker
from boto.s3.prefix import Prefix
from boto.s3.connection import S3Connection
from boto.exception import S3ResponseError
def test_delete_illegal(self):
    result = self.bucket.delete_keys([{'dict': 'notallowed'}])
    self.assertEqual(len(result.deleted), 0)
    self.assertEqual(len(result.errors), 1)