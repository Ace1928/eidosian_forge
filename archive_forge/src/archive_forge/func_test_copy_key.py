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
def test_copy_key(self):
    """Test copying a key from one bucket to another."""
    bucket1 = self._MakeBucket()
    bucket2 = self._MakeBucket()
    bucket_name_1 = bucket1.name
    bucket_name_2 = bucket2.name
    bucket1 = self._GetConnection().get_bucket(bucket_name_1)
    bucket2 = self._GetConnection().get_bucket(bucket_name_2)
    key_name = 'foobar'
    k1 = bucket1.new_key(key_name)
    self.assertIsInstance(k1, bucket1.key_class)
    k1.name = key_name
    s = 'This is a test.'
    k1.set_contents_from_string(s)
    k1.copy(bucket_name_2, key_name)
    k2 = bucket2.lookup(key_name)
    self.assertIsInstance(k2, bucket2.key_class)
    tmpdir = self._MakeTempDir()
    fpath = os.path.join(tmpdir, 'foobar')
    fp = open(fpath, 'wb')
    k2.get_contents_to_file(fp)
    fp.close()
    fp = open(fpath)
    self.assertEqual(s, fp.read())
    fp.close()
    bucket1.delete_key(k1)
    bucket2.delete_key(k2)