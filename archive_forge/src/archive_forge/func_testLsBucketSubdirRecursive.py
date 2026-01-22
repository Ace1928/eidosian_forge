from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import gzip
import os
import six
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import ServiceException
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.exception import NO_URLS_MATCHED_GENERIC
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetDummyProjectForUnitTest
from gslib.tests.util import unittest
from gslib.utils.constants import UTF8
from gslib.utils import copy_helper
from gslib.utils import system_util
def testLsBucketSubdirRecursive(self):
    """Test that ls -R of a bucket subdir returns expected results."""
    src_bucket_uri = self.CreateBucket(test_objects=['src_subdir/foo', 'src_subdir/nested/foo2'])
    for final_char in ('/', ''):
        output = self.RunCommand('ls', ['-R', suri(src_bucket_uri, 'src_subdir') + final_char], return_stdout=True)
        expected = set([suri(src_bucket_uri, 'src_subdir', ':'), suri(src_bucket_uri, 'src_subdir', 'foo'), suri(src_bucket_uri, 'src_subdir', 'nested', ':'), suri(src_bucket_uri, 'src_subdir', 'nested', 'foo2')])
        expected.add('')
        actual = set([line.strip() for line in output.split('\n')])
        self.assertEqual(expected, actual)