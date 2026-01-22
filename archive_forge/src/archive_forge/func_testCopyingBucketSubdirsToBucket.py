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
def testCopyingBucketSubdirsToBucket(self):
    """Ensure wildcarded recursive cp in bucket subdirs behaves like Unix."""
    src_bucket_uri = self.CreateBucket()
    dst_bucket_uri = self.CreateBucket()
    fpath = self.CreateTempFile(file_name='foo', contents=b'bar')
    self.RunCommand('cp', [fpath, suri(src_bucket_uri, 'Test/sub-test/foo')])
    self.RunCommand('cp', [fpath, suri(src_bucket_uri, 'Test2/sub-test/foo')])
    self.RunCommand('cp', [fpath, suri(src_bucket_uri, 'Test3/sub-test/foo')])
    self.RunCommand('cp', ['-R', suri(src_bucket_uri, '*', 'sub-test'), suri(dst_bucket_uri)])
    actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
    expected = set([suri(dst_bucket_uri, 'sub-test', 'foo')])
    self.assertEqual(expected, actual)
    src_bucket_uri2 = self.CreateBucket()
    dst_bucket_uri2 = self.CreateBucket()
    self.RunCommand('cp', [fpath, suri(src_bucket_uri2, 'Test/dir1/dir2/foo')])
    self.RunCommand('cp', [fpath, suri(src_bucket_uri2, 'Test2/dir1/dir2/foo')])
    self.RunCommand('cp', [fpath, suri(src_bucket_uri2, 'Test3/dir1/dir2/bar')])
    self.RunCommand('cp', ['-R', suri(src_bucket_uri2, '*', 'dir1', 'dir2'), suri(dst_bucket_uri2)])
    actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri2, '**')).IterAll(expand_top_level_buckets=True)))
    expected = set([suri(dst_bucket_uri2, 'dir2', 'foo'), suri(dst_bucket_uri2, 'dir2', 'bar')])
    self.assertEqual(expected, actual)
    dst_bucket_uri3 = self.CreateBucket()
    self.RunCommand('cp', ['-R', suri(src_bucket_uri2, 'Test*', '*', 'dir2'), suri(dst_bucket_uri3)])
    actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri3, '**')).IterAll(expand_top_level_buckets=True)))
    expected = set([suri(dst_bucket_uri3, 'dir2', 'foo'), suri(dst_bucket_uri3, 'dir2', 'bar')])
    self.assertEqual(expected, actual)
    src_bucket_uri3 = self.CreateBucket()
    dst_bucket_uri4 = self.CreateBucket()
    self.RunCommand('cp', [fpath, suri(src_bucket_uri3, 'dir1/test1/dir2/dir3/foo')])
    self.RunCommand('cp', [fpath, suri(src_bucket_uri3, 'dir1/test2/dir2/dir3/foo')])
    self.RunCommand('cp', [fpath, suri(src_bucket_uri3, 'dir1/test3/dir2/dir3/bar')])
    self.RunCommand('cp', ['-R', suri(src_bucket_uri3, 'dir1', '*', 'dir2', 'dir3'), suri(dst_bucket_uri4)])
    actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri4, '**')).IterAll(expand_top_level_buckets=True)))
    expected = set([suri(dst_bucket_uri4, 'dir3', 'foo'), suri(dst_bucket_uri4, 'dir3', 'bar')])
    self.assertEqual(expected, actual)