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
def test_list_iterator(self):
    """Test list and iterator."""
    bucket = self._MakeBucket()
    num_iter = len([k for k in bucket.list()])
    rs = bucket.get_all_keys()
    num_keys = len(rs)
    self.assertEqual(num_iter, num_keys)