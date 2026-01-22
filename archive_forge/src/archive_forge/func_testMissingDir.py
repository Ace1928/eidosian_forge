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
def testMissingDir(self):
    """Tests that wildcard gets empty iterator when directory doesn't exist."""
    res = list(self._test_wildcard_iterator(suri('no_such_dir', '*')).IterAll(expand_top_level_buckets=True))
    self.assertEqual(0, len(res))