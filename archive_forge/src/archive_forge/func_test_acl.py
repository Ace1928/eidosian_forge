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
def test_acl(self):
    """Test bucket and key ACLs."""
    bucket = self._MakeBucket()
    bucket.set_acl('public-read')
    acl = bucket.get_acl()
    self.assertEqual(len(acl.entries.entry_list), 2)
    bucket.set_acl('private')
    acl = bucket.get_acl()
    self.assertEqual(len(acl.entries.entry_list), 1)
    k = self._MakeKey(bucket=bucket)
    k.set_acl('public-read')
    acl = k.get_acl()
    self.assertEqual(len(acl.entries.entry_list), 2)
    k.set_acl('private')
    acl = k.get_acl()
    self.assertEqual(len(acl.entries.entry_list), 1)
    acl_xml = '<ACCESSControlList><EntrIes><Entry>' + '<Scope type="AllUsers"></Scope><Permission>READ</Permission>' + '</Entry></EntrIes></ACCESSControlList>'
    acl = ACL()
    h = handler.XmlHandler(acl, bucket)
    xml.sax.parseString(acl_xml, h)
    bucket.set_acl(acl)
    self.assertEqual(len(acl.entries.entry_list), 1)
    aclstr = k.get_xml_acl()
    self.assertGreater(aclstr.count('/Entry', 1), 0)