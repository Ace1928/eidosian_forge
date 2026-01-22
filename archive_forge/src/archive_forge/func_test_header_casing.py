from tests.unit import unittest
import time
import random
import boto.s3
from boto.compat import six, StringIO, urllib
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3ResponseError
def test_header_casing(self):
    key = self.bucket.new_key('test_header_case')
    key.set_metadata('Content-type', 'application/json')
    key.set_metadata('Content-md5', 'XmUKnus7svY1frWsVskxXg==')
    key.set_contents_from_string('{"abc": 123}')
    check = self.bucket.get_key('test_header_case')
    self.assertEqual(check.content_type, 'application/json')