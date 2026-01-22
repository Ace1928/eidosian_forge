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
def testCopyingOneNestedFileToBucketSubDir(self):
    """Tests copying one nested file to a bucket subdir."""
    dst_bucket_uri = self.CreateBucket(test_objects=['d0/placeholder', 'd1/placeholder'])
    src_dir = self.CreateTempDir(test_files=[('d3', 'd4', 'nested', 'f1')])
    for i, final_dst_char in enumerate(('', '/')):
        self.RunCommand('cp', ['-r', suri(src_dir, 'd3'), suri(dst_bucket_uri, 'd%d' % i) + final_dst_char])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
    expected = set([suri(dst_bucket_uri, 'd0', 'placeholder'), suri(dst_bucket_uri, 'd1', 'placeholder'), suri(dst_bucket_uri, 'd0', 'd3', 'd4', 'nested', 'f1'), suri(dst_bucket_uri, 'd1', 'd3', 'd4', 'nested', 'f1')])
    self.assertEqual(expected, actual)