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
def test_default_object_acls(self):
    """Test default object acls."""
    bucket = self._MakeBucket()
    acl = bucket.get_def_acl()
    self.assertIsNotNone(re.search(PROJECT_PRIVATE_RE, acl.to_xml()))
    bucket.set_def_acl('public-read')
    acl = bucket.get_def_acl()
    public_read_acl = acl
    self.assertEqual(acl.to_xml(), '<AccessControlList><Entries><Entry><Scope type="AllUsers"></Scope><Permission>READ</Permission></Entry></Entries></AccessControlList>')
    bucket.set_def_acl('private')
    acl = bucket.get_def_acl()
    self.assertEqual(acl.to_xml(), '<AccessControlList></AccessControlList>')
    bucket.set_def_acl(public_read_acl)
    acl = bucket.get_def_acl()
    self.assertEqual(acl.to_xml(), '<AccessControlList><Entries><Entry><Scope type="AllUsers"></Scope><Permission>READ</Permission></Entry></Entries></AccessControlList>')
    bucket.set_def_acl('private')
    acl = bucket.get_def_acl()
    self.assertEqual(acl.to_xml(), '<AccessControlList></AccessControlList>')