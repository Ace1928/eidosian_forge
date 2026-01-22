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
def testWildcardFields(self):
    """Tests that wildcard w/fields specification returns correct fields."""
    blrs = set((u for u in self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('**')).IterAll(bucket_listing_fields=['timeCreated'])))
    self.assertTrue(len(blrs))
    for blr in blrs:
        self.assertTrue(blr.root_object and blr.root_object.timeCreated)
    blrs = set((u for u in self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('**')).IterAll(bucket_listing_fields=['generation'])))
    self.assertTrue(len(blrs))
    for blr in blrs:
        self.assertTrue(blr.root_object and (not blr.root_object.timeCreated))