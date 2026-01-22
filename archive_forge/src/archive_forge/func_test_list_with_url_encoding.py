from mock import patch, Mock
import unittest
import time
from boto.exception import S3ResponseError
from boto.s3.connection import S3Connection
from boto.s3.bucketlogging import BucketLogging
from boto.s3.lifecycle import Lifecycle
from boto.s3.lifecycle import Transition
from boto.s3.lifecycle import Expiration
from boto.s3.lifecycle import Rule
from boto.s3.acl import Grant
from boto.s3.tagging import Tags, TagSet
from boto.s3.website import RedirectLocation
from boto.compat import unquote_str
def test_list_with_url_encoding(self):
    expected = [u'α', u'β', u'γ']
    for key_name in expected:
        key = self.bucket.new_key(key_name)
        key.set_contents_from_string(key_name)
    orig_getall = self.bucket._get_all
    getall = lambda *a, **k: orig_getall(*a, max_keys=2, **k)
    with patch.object(self.bucket, '_get_all', getall):
        rs = self.bucket.list(encoding_type='url')
        for element in rs:
            name = unquote_str(element.name)
            self.assertEqual(name, expected.pop(0))
        self.assertEqual(expected, [])