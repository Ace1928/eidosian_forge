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
def testMultiMatchWildcardedBucketUri(self):
    """Tests matching a multiple buckets based on a wildcarded bucket URI."""
    exp_obj_uri_strs = set([suri(self.test_bucket0_uri) + self.test_bucket0_uri.delim, suri(self.test_bucket1_uri) + self.test_bucket1_uri.delim, suri(self.test_bucket2_uri) + self.test_bucket2_uri.delim])
    with SetDummyProjectForUnitTest():
        actual_obj_uri_strs = set((str(u) for u in self._test_wildcard_iterator('%s*' % self.base_uri_str).IterBuckets(bucket_fields=['id'])))
    self.assertEqual(exp_obj_uri_strs, actual_obj_uri_strs)