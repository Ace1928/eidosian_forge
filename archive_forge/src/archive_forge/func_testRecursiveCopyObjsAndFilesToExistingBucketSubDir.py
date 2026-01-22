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
def testRecursiveCopyObjsAndFilesToExistingBucketSubDir(self):
    """Tests recursive copy of objects and files to existing bucket subdir."""
    src_bucket_uri = self.CreateBucket(test_objects=['f0', 'nested/f1'])
    dst_bucket_uri = self.CreateBucket(test_objects=['dst_subdir0/existing_obj', 'dst_subdir1/existing_obj'])
    src_dir = self.CreateTempDir(test_files=['f2', ('nested', 'f3')])
    for i, final_char in enumerate(('/', '')):
        self.RunCommand('cp', ['-R', suri(src_bucket_uri), src_dir, suri(dst_bucket_uri, 'dst_subdir%d' % i) + final_char])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, 'dst_subdir%d' % i, '**')).IterAll(expand_top_level_buckets=True)))
        tmp_dirname = os.path.split(src_dir)[1]
        bucketname = src_bucket_uri.bucket_name
        expected = set([suri(dst_bucket_uri, 'dst_subdir%d' % i, 'existing_obj'), suri(dst_bucket_uri, 'dst_subdir%d' % i, bucketname, 'f0'), suri(dst_bucket_uri, 'dst_subdir%d' % i, bucketname, 'nested', 'f1'), suri(dst_bucket_uri, 'dst_subdir%d' % i, tmp_dirname, 'f2'), suri(dst_bucket_uri, 'dst_subdir%d' % i, tmp_dirname, 'nested', 'f3')])
        self.assertEqual(expected, actual)