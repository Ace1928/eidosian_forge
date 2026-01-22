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
def testFlatCopyingObjsAndFilesToBucketSubDir(self):
    """Tests copying flatly listed objects and files to bucket subdir."""
    src_bucket_uri = self.CreateBucket(test_objects=['f0', 'd0/f1', 'd1/d2/f2'])
    src_dir = self.CreateTempDir(test_files=['f3', ('d3', 'f4'), ('d4', 'd5', 'f5')])
    dst_bucket_uri = self.CreateBucket(test_objects=['dst_subdir0/existing', 'dst_subdir1/existing'])
    for i, final_char in enumerate(('/', '')):
        self.RunCommand('cp', ['-R', suri(src_bucket_uri, '**'), os.path.join(src_dir, '**'), suri(dst_bucket_uri, 'dst_subdir%d' % i) + final_char])
    actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
    expected = set()
    for i in range(2):
        expected.add(suri(dst_bucket_uri, 'dst_subdir%d' % i, 'existing'))
        for j in range(6):
            expected.add(suri(dst_bucket_uri, 'dst_subdir%d' % i, 'f%d' % j))
    self.assertEqual(expected, actual)