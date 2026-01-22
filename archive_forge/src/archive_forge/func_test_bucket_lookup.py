import os
import re
import urllib
import xml.sax
from six import StringIO
from boto import handler
from boto import storage_uri
from boto.gs.acl import ACL
from boto.gs.cors import Cors
from boto.gs.lifecycle import LifecycleConfig
from tests.integration.gs.testcase import GSTestCase
def test_bucket_lookup(self):
    """Test the bucket lookup method."""
    bucket = self._MakeBucket()
    k = bucket.new_key('foo/bar')
    phony_mimetype = 'application/x-boto-test'
    headers = {'Content-Type': phony_mimetype}
    k.set_contents_from_string('testdata', headers)
    k = bucket.lookup('foo/bar')
    self.assertIsInstance(k, bucket.key_class)
    self.assertEqual(k.content_type, phony_mimetype)
    k = bucket.lookup('notthere')
    self.assertIsNone(k)