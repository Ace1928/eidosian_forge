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
def testInvalidRecursiveDirectoryWildcard(self):
    """Tests that wildcard containing '***' raises exception."""
    try:
        uri = self._test_storage_uri(suri(self.test_dir, '***', 'abcd'))
        for unused_ in self._test_wildcard_iterator(uri).IterAll(expand_top_level_buckets=True):
            self.fail('Expected WildcardException not raised.')
    except wildcard_iterator.WildcardException as e:
        self.assertTrue(str(e).find('more than 2 consecutive') != -1)