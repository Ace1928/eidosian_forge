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
def test_bucket_to_bucket_src_mtime(self):
    """Tests bucket to bucket where source has mtime in files."""
    src_bucket = self.CreateBucket()
    dst_bucket = self.CreateBucket()
    obj1 = self.CreateObject(bucket_uri=src_bucket, object_name='obj1', contents=b'obj1', mtime=0)
    obj2 = self.CreateObject(bucket_uri=src_bucket, object_name='subdir/obj2', contents=b'subdir/obj2', mtime=1)
    self._VerifyObjectMtime(obj1.bucket_name, obj1.object_name, '0')
    self._VerifyObjectMtime(obj2.bucket_name, obj2.object_name, '1')

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        """Tests rsync works as expected."""
        self.RunGsUtil(['rsync', '-r', suri(src_bucket), suri(dst_bucket)])
        listing1 = TailSet(suri(src_bucket), self.FlatListBucket(src_bucket))
        listing2 = TailSet(suri(dst_bucket), self.FlatListBucket(dst_bucket))
        self.assertEqual(listing1, set(['/obj1', '/subdir/obj2']))
        self.assertEqual(listing2, set(['/obj1', '/subdir/obj2']))
    _Check1()
    self._VerifyObjectMtime(dst_bucket.bucket_name, 'obj1', '0')
    self._VerifyObjectMtime(dst_bucket.bucket_name, 'subdir/obj2', '1')