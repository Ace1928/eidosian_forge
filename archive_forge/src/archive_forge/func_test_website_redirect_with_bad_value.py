from tests.unit import unittest
import time
import random
import boto.s3
from boto.compat import six, StringIO, urllib
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3ResponseError
def test_website_redirect_with_bad_value(self):
    self.bucket.configure_website('index.html')
    key = self.bucket.new_key('redirect-key')
    with self.assertRaises(key.provider.storage_response_error):
        key.set_redirect('ftp://ftp.example.org')
    with self.assertRaises(key.provider.storage_response_error):
        key.set_redirect('')