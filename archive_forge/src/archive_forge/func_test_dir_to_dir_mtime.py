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
@unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
def test_dir_to_dir_mtime(self):
    """Tests that flat and recursive rsync dir to dir works correctly."""
    tmpdir1 = self.CreateTempDir()
    tmpdir2 = self.CreateTempDir()
    subdir1 = os.path.join(tmpdir1, 'subdir1')
    subdir2 = os.path.join(tmpdir2, 'subdir2')
    os.mkdir(subdir1)
    os.mkdir(subdir2)
    self.CreateTempFile(tmpdir=tmpdir1, file_name='obj1', contents=b'obj1', mtime=10)
    self.CreateTempFile(tmpdir=tmpdir1, file_name='.obj2', contents=b'.obj2', mtime=10)
    self.CreateTempFile(tmpdir=subdir1, file_name='obj3', contents=b'subdir1/obj3', mtime=10)
    self.CreateTempFile(tmpdir=tmpdir1, file_name='obj6', contents=b'obj6', mtime=100)
    self.CreateTempFile(tmpdir=tmpdir1, file_name='obj7', contents=b'obj7_', mtime=100)
    self.CreateTempFile(tmpdir=tmpdir2, file_name='.obj2', contents=b'.OBJ2', mtime=1000)
    self.CreateTempFile(tmpdir=tmpdir2, file_name='obj4', contents=b'obj4', mtime=10)
    self.CreateTempFile(tmpdir=subdir2, file_name='obj5', contents=b'subdir2/obj5', mtime=10)
    self.CreateTempFile(tmpdir=tmpdir2, file_name='obj6', contents=b'OBJ6', mtime=100)
    self.CreateTempFile(tmpdir=tmpdir2, file_name='obj7', contents=b'obj7', mtime=100)
    self.RunGsUtil(['rsync', '-r', '-d', tmpdir1, tmpdir2])
    listing1 = TailSet(tmpdir1, self.FlatListDir(tmpdir1))
    listing2 = TailSet(tmpdir2, self.FlatListDir(tmpdir2))
    self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir1/obj3', '/obj6', '/obj7']))
    self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir1/obj3', '/obj6', '/obj7']))
    with open(os.path.join(tmpdir2, '.obj2')) as f:
        self.assertEqual('.obj2', '\n'.join(f.readlines()))
    with open(os.path.join(tmpdir2, 'obj6')) as f:
        self.assertEqual('OBJ6', '\n'.join(f.readlines()))
    with open(os.path.join(tmpdir2, 'obj7')) as f:
        self.assertEqual('obj7_', '\n'.join(f.readlines()))

    def _Check1():
        self.assertEqual(NO_CHANGES, self.RunGsUtil(['rsync', '-d', tmpdir1, tmpdir2], return_stderr=True))
    _Check1()
    self.RunGsUtil(['rsync', '-r', '-d', '-c', tmpdir1, tmpdir2])
    listing1 = TailSet(tmpdir1, self.FlatListDir(tmpdir1))
    listing2 = TailSet(tmpdir2, self.FlatListDir(tmpdir2))
    self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir1/obj3', '/obj6', '/obj7']))
    self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir1/obj3', '/obj6', '/obj7']))
    with open(os.path.join(tmpdir1, '.obj2')) as f:
        self.assertEqual('.obj2', '\n'.join(f.readlines()))
    with open(os.path.join(tmpdir1, '.obj2')) as f:
        self.assertEqual('.obj2', '\n'.join(f.readlines()))
    with open(os.path.join(tmpdir2, 'obj6')) as f:
        self.assertEqual('obj6', '\n'.join(f.readlines()))

    def _Check2():
        self.assertEqual(NO_CHANGES, self.RunGsUtil(['rsync', '-d', '-c', tmpdir1, tmpdir2], return_stderr=True))
    _Check2()
    os.unlink(os.path.join(tmpdir1, 'obj7'))
    os.unlink(os.path.join(tmpdir2, 'obj7'))
    self.CreateTempFile(tmpdir=tmpdir1, file_name='obj6', contents=b'obj6', mtime=10)
    self.CreateTempFile(tmpdir=tmpdir2, file_name='obj7', contents=b'obj7', mtime=100)
    os.unlink(os.path.join(tmpdir1, 'obj1'))
    os.unlink(os.path.join(tmpdir2, '.obj2'))
    self.RunGsUtil(['rsync', '-d', '-r', tmpdir1, tmpdir2])
    listing1 = TailSet(tmpdir1, self.FlatListDir(tmpdir1))
    listing2 = TailSet(tmpdir2, self.FlatListDir(tmpdir2))
    self.assertEqual(listing1, set(['/.obj2', '/obj6', '/subdir1/obj3']))
    self.assertEqual(listing2, set(['/.obj2', '/obj6', '/subdir1/obj3']))

    def _Check3():
        self.assertEqual(NO_CHANGES, self.RunGsUtil(['rsync', '-d', '-r', tmpdir1, tmpdir2], return_stderr=True))
    _Check3()