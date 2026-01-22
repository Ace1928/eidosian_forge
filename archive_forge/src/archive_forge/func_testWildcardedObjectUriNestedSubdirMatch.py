from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import six
import tempfile
from gslib import wildcard_iterator
from gslib.exception import InvalidUrlError
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetDummyProjectForUnitTest
def testWildcardedObjectUriNestedSubdirMatch(self):
    """Tests wildcarding with a nested subdir."""
    uri_strs = set()
    prefixes = set()
    for blr in self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('*')):
        if blr.IsPrefix():
            prefixes.add(blr.root_object)
        else:
            uri_strs.add(blr.url_string)
    exp_obj_uri_strs = set([suri(self.test_bucket0_uri, x) for x in self.immed_child_obj_names])
    self.assertEqual(exp_obj_uri_strs, uri_strs)
    self.assertEqual(1, len(prefixes))
    self.assertTrue('nested1/' in prefixes)