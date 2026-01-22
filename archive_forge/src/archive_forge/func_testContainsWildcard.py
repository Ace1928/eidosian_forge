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
def testContainsWildcard(self):
    """Tests ContainsWildcard call."""
    self.assertTrue(ContainsWildcard('a*.txt'))
    self.assertTrue(ContainsWildcard('a[0-9].txt'))
    self.assertFalse(ContainsWildcard('0-9.txt'))
    self.assertTrue(ContainsWildcard('?.txt'))