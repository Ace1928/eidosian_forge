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
def test_billing_config_storage_uri(self):
    """Test setting and getting of billing config with storage_uri."""
    bucket = self._MakeBucket()
    bucket_name = bucket.name
    uri = storage_uri('gs://' + bucket_name)
    billing = uri.get_billing_config()
    self.assertEqual(billing, BILLING_EMPTY)
    uri.configure_billing(requester_pays=True)
    billing = uri.get_billing_config()
    self.assertEqual(billing, BILLING_ENABLED)
    uri.configure_billing(requester_pays=False)
    billing = uri.get_billing_config()
    self.assertEqual(billing, BILLING_DISABLED)