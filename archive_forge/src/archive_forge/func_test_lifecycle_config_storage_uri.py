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
def test_lifecycle_config_storage_uri(self):
    """Test setting and getting of lifecycle config with storage_uri."""
    bucket = self._MakeBucket()
    bucket_name = bucket.name
    uri = storage_uri('gs://' + bucket_name)
    xml = uri.get_lifecycle_config().to_xml()
    self.assertEqual(xml, LIFECYCLE_EMPTY)
    lifecycle_config = LifecycleConfig()
    lifecycle_config.add_rule('Delete', None, LIFECYCLE_CONDITIONS_FOR_DELETE_RULE)
    lifecycle_config.add_rule('SetStorageClass', 'NEARLINE', LIFECYCLE_CONDITIONS_FOR_SET_STORAGE_CLASS_RULE)
    uri.configure_lifecycle(lifecycle_config)
    xml = uri.get_lifecycle_config().to_xml()
    self.assertEqual(xml, LIFECYCLE_DOC)