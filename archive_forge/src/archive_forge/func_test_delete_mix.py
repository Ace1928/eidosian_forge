import unittest
import time
from boto.s3.key import Key
from boto.s3.deletemarker import DeleteMarker
from boto.s3.prefix import Prefix
from boto.s3.connection import S3Connection
from boto.exception import S3ResponseError
def test_delete_mix(self):
    result = self.bucket.delete_keys(['king', ('mice', None), Key(name='regular'), Key(), Prefix(name='folder/'), DeleteMarker(name='deleted'), {'bad': 'type'}])
    self.assertEqual(len(result.deleted), 4)
    self.assertEqual(len(result.errors), 3)