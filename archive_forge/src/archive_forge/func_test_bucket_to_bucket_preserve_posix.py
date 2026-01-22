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
@unittest.skipIf(IS_WINDOWS, 'POSIX attributes not available on Windows.')
@unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
def test_bucket_to_bucket_preserve_posix(self):
    """Tests that rsync -P works with bucket to bucket."""
    src_bucket = self.CreateBucket()
    dst_bucket = self.CreateBucket()
    primary_gid = os.getgid()
    non_primary_gid = util.GetNonPrimaryGid()
    self.CreateObject(bucket_uri=src_bucket, object_name='obj1', contents=b'obj1', mode='444')
    self.CreateObject(bucket_uri=src_bucket, object_name='obj2', contents=b'obj2', gid=primary_gid)
    self.CreateObject(bucket_uri=src_bucket, object_name='obj3', contents=b'obj3', gid=non_primary_gid)
    self.CreateObject(bucket_uri=src_bucket, object_name='obj4', contents=b'obj3', uid=INVALID_UID(), gid=INVALID_GID(), mode='222')
    self.CreateObject(bucket_uri=src_bucket, object_name='obj5', contents=b'obj5', uid=USER_ID, gid=primary_gid, mode=str(DEFAULT_MODE))
    self.CreateObject(bucket_uri=dst_bucket, object_name='obj5', contents=b'obj5')

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        """Test bucket to bucket rsync with -P flag and verify attributes."""
        stderr = self.RunGsUtil(['rsync', '-P', suri(src_bucket), suri(dst_bucket)], return_stderr=True)
        listing1 = TailSet(suri(src_bucket), self.FlatListBucket(src_bucket))
        listing2 = TailSet(suri(dst_bucket), self.FlatListBucket(dst_bucket))
        self.assertEqual(listing1, set(['/obj1', '/obj2', '/obj3', '/obj4', '/obj5']))
        self.assertEqual(listing2, set(['/obj1', '/obj2', '/obj3', '/obj4', '/obj5']))
        if self._use_gcloud_storage:
            self.assertIn('Patching', stderr)
        else:
            self.assertIn('Copying POSIX attributes from src to dst for', stderr)
    _Check1()
    self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj1', MODE_ATTR, '444')
    self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj2', GID_ATTR, str(primary_gid))
    self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj3', GID_ATTR, str(non_primary_gid))
    self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj4', GID_ATTR, str(INVALID_GID()))
    self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj4', UID_ATTR, str(INVALID_UID()))
    self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj4', MODE_ATTR, '222')
    self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj5', UID_ATTR, str(USER_ID))
    self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj5', GID_ATTR, str(primary_gid))
    self.VerifyObjectCustomAttribute(dst_bucket.bucket_name, 'obj5', MODE_ATTR, str(DEFAULT_MODE))

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check2():
        """Check that we are not patching destination metadata a second time."""
        stderr = self.RunGsUtil(['rsync', '-P', suri(src_bucket), suri(dst_bucket)], return_stderr=True)
        if self._use_gcloud_storage:
            self.assertNotIn('Patching', stderr)
        else:
            self.assertNotIn('Copying POSIX attributes from src to dst for', stderr)
    _Check2()