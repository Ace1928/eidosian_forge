import binascii
import re
from six import StringIO
from boto import storage_uri
from boto.exception import BotoClientError
from boto.gs.acl import SupportedPermissions as perms
from tests.integration.gs.testcase import GSTestCase
def testCloneReplaceKey(self):
    b = self._MakeBucket()
    k = b.new_key('obj')
    k.set_contents_from_string('stringdata')
    orig_uri = storage_uri('gs://%s/' % b.name)
    uri = orig_uri.clone_replace_key(k)
    self.assertTrue(uri.has_version())
    self.assertRegexpMatches(str(uri.generation), '[0-9]+')