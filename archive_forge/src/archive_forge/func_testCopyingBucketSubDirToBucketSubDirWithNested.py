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
def testCopyingBucketSubDirToBucketSubDirWithNested(self):
    """Tests copying a bucket subdir to another bucket subdir with nesting."""
    src_bucket_uri = self.CreateBucket(test_objects=['src_subdir_%d/obj' % i for i in range(4)] + ['src_subdir_%d/nested/obj' % i for i in range(4)])
    dst_bucket_uri = self.CreateBucket(test_objects=['dst_subdir_%d/obj2' % i for i in range(4)])
    for i, (final_src_char, final_dst_char) in enumerate((('', ''), ('', '/'), ('/', ''), ('/', '/'))):
        self.RunCommand('cp', ['-R', suri(src_bucket_uri, 'src_subdir_%d' % i) + final_src_char, suri(dst_bucket_uri, 'dst_subdir_%d' % i) + final_dst_char])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, 'dst_subdir_%d' % i, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'dst_subdir_%d' % i, 'src_subdir_%d' % i, 'obj'), suri(dst_bucket_uri, 'dst_subdir_%d' % i, 'src_subdir_%d' % i, 'nested', 'obj'), suri(dst_bucket_uri, 'dst_subdir_%d' % i, 'obj2')])
        self.assertEqual(expected, actual)