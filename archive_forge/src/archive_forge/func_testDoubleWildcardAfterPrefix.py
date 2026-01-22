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
def testDoubleWildcardAfterPrefix(self):
    """Tests gs://bucket/dir/**/object matching."""
    actual_uri_strs = set()
    actual_prefixes = set()
    for blr in self._test_wildcard_iterator(self.test_bucket2_uri.clone_replace_name('double/**/f.txt')):
        if blr.IsPrefix():
            actual_prefixes.add(blr.root_object)
        else:
            actual_uri_strs.add(blr.url_string)
    expected_uri_strs = set([self.test_bucket2_uri.clone_replace_name('double/f.txt').uri, self.test_bucket2_uri.clone_replace_name('double/foo/f.txt').uri, self.test_bucket2_uri.clone_replace_name('double/bar/f.txt').uri])
    expected_prefixes = set()
    self.assertEqual(expected_prefixes, actual_prefixes)
    self.assertEqual(expected_uri_strs, actual_uri_strs)