import os
import tempfile
from xml import sax
from six import StringIO
from boto import handler
from boto.exception import GSResponseError
from boto.gs.acl import ACL
from tests.integration.gs.testcase import GSTestCase
def testConditionalSetContentsFromString(self):
    b = self._MakeBucket()
    k = b.new_key('foo')
    s1 = 'test1'
    with self.assertRaisesRegexp(GSResponseError, VERSION_MISMATCH):
        k.set_contents_from_string(s1, if_generation=999)
    k.set_contents_from_string(s1, if_generation=0)
    g1 = k.generation
    s2 = 'test2'
    with self.assertRaisesRegexp(GSResponseError, VERSION_MISMATCH):
        k.set_contents_from_string(s2, if_generation=int(g1) + 1)
    k.set_contents_from_string(s2, if_generation=g1)
    self.assertEqual(k.get_contents_as_string(), s2)