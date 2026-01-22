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
@SequentialAndParallelTransfer
@unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
def test_dir_to_bucket_minus_d(self):
    """Tests that flat and recursive rsync dir to bucket works correctly."""
    tmpdir = self.CreateTempDir()
    subdir = os.path.join(tmpdir, 'subdir')
    os.mkdir(subdir)
    bucket_uri = self.CreateBucket()
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1')
    o2_path = self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.obj2')
    self.CreateTempFile(tmpdir=subdir, file_name='obj3', contents=b'subdir/obj3')
    self.CreateObject(bucket_uri=bucket_uri, object_name='.obj2', contents=b'.OBJ2')
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj4', contents=b'obj4')
    self.CreateObject(bucket_uri=bucket_uri, object_name='subdir/obj5', contents=b'subdir/obj5')
    self.AssertNObjectsInBucket(bucket_uri, 3)

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        """Tests rsync works as expected."""
        self.RunGsUtil(['rsync', '-d', tmpdir, suri(bucket_uri)])
        listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3']))
        self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj5']))
        with open(os.path.join(tmpdir, '.obj2')) as f:
            self.assertEqual('.obj2', '\n'.join(f.readlines()))
        cloud_obj2_content = self.RunGsUtil(['cat', suri(bucket_uri, '.obj2')], return_stdout=True)
        if self._use_gcloud_storage:
            local_obj2_mtime = int(os.path.getmtime(o2_path))
            cloud_obj2_ctime = ConvertDatetimeToPOSIX(self._GetMetadataAttribute(bucket_uri.bucket_name, '.obj2', 'timeCreated'))
            self.assertTrue(cloud_obj2_content == '.obj2' or local_obj2_mtime == cloud_obj2_ctime)
        else:
            self.assertEqual('.obj2', cloud_obj2_content)
    _Check1()

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check2():
        stderr = self.RunGsUtil(['rsync', '-d', tmpdir, suri(bucket_uri)], return_stderr=True)
        self._VerifyNoChanges(stderr)
    _Check2()

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check3():
        """Tests rsync -c works as expected."""
        self.RunGsUtil(['rsync', '-d', '-c', tmpdir, suri(bucket_uri)])
        listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3']))
        self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj5']))
        with open(os.path.join(tmpdir, '.obj2')) as f:
            self.assertEqual('.obj2', '\n'.join(f.readlines()))
        self.assertEqual('.obj2', self.RunGsUtil(['cat', suri(bucket_uri, '.obj2')], return_stdout=True))
    _Check3()

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check4():
        stderr = self.RunGsUtil(['rsync', '-d', '-c', tmpdir, suri(bucket_uri)], return_stderr=True)
        self._VerifyNoChanges(stderr)
    _Check4()
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj6', contents=b'obj6')
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj7', contents=b'obj7')
    os.unlink(os.path.join(tmpdir, 'obj1'))
    self.RunGsUtil(['rm', suri(bucket_uri, '.obj2')])

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check5():
        self.RunGsUtil(['rsync', '-d', '-r', tmpdir, suri(bucket_uri)])
        listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        self.assertEqual(listing1, set(['/.obj2', '/obj6', '/subdir/obj3']))
        self.assertEqual(listing2, set(['/.obj2', '/obj6', '/subdir/obj3']))
    _Check5()

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check6():
        stderr = self.RunGsUtil(['rsync', '-d', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
        self._VerifyNoChanges(stderr)
    _Check6()