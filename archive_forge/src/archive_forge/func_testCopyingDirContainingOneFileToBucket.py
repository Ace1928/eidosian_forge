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
def testCopyingDirContainingOneFileToBucket(self):
    """Tests copying a directory containing 1 file to a bucket.

    We test this case to ensure that correct bucket handling isn't dependent
    on the copy being treated as a multi-source copy.
    """
    dst_bucket_uri = self.CreateBucket()
    src_dir = self.CreateTempDir(test_files=[('dir0', 'dir1', 'foo')])
    self.RunCommand('cp', ['-R', os.path.join(src_dir, 'dir0', 'dir1'), suri(dst_bucket_uri)])
    actual = list((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
    self.assertEqual(1, len(actual))
    self.assertEqual(suri(dst_bucket_uri, 'dir1', 'foo'), actual[0])