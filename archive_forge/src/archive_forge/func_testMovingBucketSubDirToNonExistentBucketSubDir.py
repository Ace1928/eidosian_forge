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
def testMovingBucketSubDirToNonExistentBucketSubDir(self):
    """Tests moving a bucket subdir to a non-existent bucket subdir."""
    src_bucket = self.CreateBucket(test_objects=['foo', 'src_subdir0/foo2', 'src_subdir0/nested/foo3', 'src_subdir1/foo2', 'src_subdir1/nested/foo3'])
    dst_bucket = self.CreateBucket()
    for i, final_src_char in enumerate(('', '/')):
        self.RunCommand('mv', [suri(src_bucket, 'src_subdir%d' % i) + final_src_char, suri(dst_bucket, 'dst_subdir%d' % i)])
    actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket, '**')).IterAll(expand_top_level_buckets=True)))
    expected = set([suri(dst_bucket, 'dst_subdir0', 'foo2'), suri(dst_bucket, 'dst_subdir1', 'foo2'), suri(dst_bucket, 'dst_subdir0', 'nested', 'foo3'), suri(dst_bucket, 'dst_subdir1', 'nested', 'foo3')])
    self.assertEqual(expected, actual)