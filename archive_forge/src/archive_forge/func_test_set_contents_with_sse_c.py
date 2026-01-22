from tests.unit import unittest
import time
import random
import boto.s3
from boto.compat import six, StringIO, urllib
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3ResponseError
def test_set_contents_with_sse_c(self):
    content = '01234567890123456789'
    header = {'x-amz-server-side-encryption-customer-algorithm': 'AES256', 'x-amz-server-side-encryption-customer-key': 'MAAxAHQAZQBzAHQASwBlAHkAVABvAFMAUwBFAEMAIQA=', 'x-amz-server-side-encryption-customer-key-MD5': 'fUgCZDDh6bfEMuP2bN38mg=='}
    k = self.bucket.new_key('testkey_for_sse_c')
    k.set_contents_from_string(content, headers=header)
    kn = self.bucket.new_key('testkey_for_sse_c')
    ks = kn.get_contents_as_string(headers=header)
    self.assertEqual(ks, content.encode('utf-8'))