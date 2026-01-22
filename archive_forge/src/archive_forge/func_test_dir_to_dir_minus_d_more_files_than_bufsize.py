from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
from unittest import mock
import six
from gslib import command
from gslib.commands import rsync
from gslib.project_id import PopulateProjectId
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import AuthorizeProjectToUseTestingKmsKey
from gslib.tests.util import TEST_ENCRYPTION_KEY_S3
from gslib.tests.util import TEST_ENCRYPTION_KEY_S3_MD5
from gslib.tests.util import BuildErrorRegex
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import ORPHANED_FILE
from gslib.tests.util import POSIX_GID_ERROR
from gslib.tests.util import POSIX_INSUFFICIENT_ACCESS_ERROR
from gslib.tests.util import POSIX_MODE_ERROR
from gslib.tests.util import POSIX_UID_ERROR
from gslib.tests.util import SequentialAndParallelTransfer
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TailSet
from gslib.tests.util import unittest
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.hashing_helper import SLOW_CRCMOD_RSYNC_WARNING
from gslib.utils.posix_util import ConvertDatetimeToPOSIX
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import MTIME_ATTR
from gslib.utils.posix_util import NA_TIME
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_OSX
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils import shim_util
def test_dir_to_dir_minus_d_more_files_than_bufsize(self):
    """Tests concurrently building listing from multiple tmp file ranges."""
    tmpdir1 = self.CreateTempDir()
    tmpdir2 = self.CreateTempDir()
    for i in range(0, 1000):
        self.CreateTempFile(tmpdir=tmpdir1, file_name='d1-%s' % i, contents=b'x', mtime=i + 1)
        self.CreateTempFile(tmpdir=tmpdir2, file_name='d2-%s' % i, contents=b'y', mtime=i)
    rsync_buffer_config = [('GSUtil', 'rsync_buffer_lines', '50' if IS_WINDOWS else '2')]
    with SetBotoConfigForTest(rsync_buffer_config):
        self.RunGsUtil(['rsync', '-d', tmpdir1, tmpdir2])
    listing1 = TailSet(tmpdir1, self.FlatListDir(tmpdir1))
    listing2 = TailSet(tmpdir2, self.FlatListDir(tmpdir2))
    self.assertEqual(listing1, listing2)
    for i in range(0, 1000):
        self.assertEqual(i + 1, long(os.path.getmtime(os.path.join(tmpdir2, 'd1-%s' % i))))
        with open(os.path.join(tmpdir2, 'd1-%s' % i)) as f:
            self.assertEqual('x', '\n'.join(f.readlines()))

    def _Check():
        self.assertEqual(NO_CHANGES, self.RunGsUtil(['rsync', '-d', tmpdir1, tmpdir2], return_stderr=True))
    _Check()