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
def testNoOpObjectIterator(self):
    """Tests that bucket-only URI iterates just that one URI."""
    results = list(self._test_wildcard_iterator(self.test_bucket0_uri).IterBuckets(bucket_fields=['id']))
    self.assertEqual(1, len(results))
    self.assertEqual(str(self.test_bucket0_uri), str(results[0]))