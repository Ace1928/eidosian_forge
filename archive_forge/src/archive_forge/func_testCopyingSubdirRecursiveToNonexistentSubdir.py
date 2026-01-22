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
def testCopyingSubdirRecursiveToNonexistentSubdir(self):
    """Tests copying a directory with a single file recursively to a bucket.

    The file should end up in a new bucket subdirectory with the file's
    directory structure starting below the recursive copy point, as in Unix cp.

    Example:
      filepath: dir1/dir2/foo
      cp -r dir1 dir3
      Results in dir3/dir2/foo being created.
    """
    src_dir = self.CreateTempDir()
    self.CreateTempFile(tmpdir=src_dir + '/dir1/dir2', file_name='foo')
    dst_bucket_uri = self.CreateBucket()
    self.RunCommand('cp', ['-R', src_dir + '/dir1', suri(dst_bucket_uri, 'dir3')])
    actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
    expected = set([suri(dst_bucket_uri, 'dir3/dir2/foo')])
    self.assertEqual(expected, actual)