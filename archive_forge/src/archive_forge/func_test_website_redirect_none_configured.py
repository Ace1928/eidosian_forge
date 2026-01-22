from tests.unit import unittest
import time
import random
import boto.s3
from boto.compat import six, StringIO, urllib
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3ResponseError
def test_website_redirect_none_configured(self):
    key = self.bucket.new_key('redirect-key')
    key.set_contents_from_string('')
    self.assertEqual(key.get_redirect(), None)