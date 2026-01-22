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
def testCopyingAbsolutePathDirToBucket(self):
    """Tests recursively copying absolute path directory to a bucket."""
    dst_bucket_uri = self.CreateBucket()
    src_dir_root = self.CreateTempDir(test_files=['f0', 'f1', 'f2.txt', ('dir0', 'dir1', 'nested')])
    self.RunCommand('cp', ['-R', src_dir_root, suri(dst_bucket_uri)])
    actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
    src_tmpdir = os.path.split(src_dir_root)[1]
    expected = set([suri(dst_bucket_uri, src_tmpdir, 'f0'), suri(dst_bucket_uri, src_tmpdir, 'f1'), suri(dst_bucket_uri, src_tmpdir, 'f2.txt'), suri(dst_bucket_uri, src_tmpdir, 'dir0', 'dir1', 'nested')])
    self.assertEqual(expected, actual)