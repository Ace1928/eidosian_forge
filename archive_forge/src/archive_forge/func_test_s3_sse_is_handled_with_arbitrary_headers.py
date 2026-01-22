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
@SkipForGS('Tests that S3 SSE-C is handled.')
def test_s3_sse_is_handled_with_arbitrary_headers(self):
    tmp_dir = self.CreateTempDir()
    tmp_file = self.CreateTempFile(tmpdir=tmp_dir, contents=b'foo')
    bucket_uri1 = self.CreateBucket()
    bucket_uri2 = self.CreateBucket()
    header_flags = ['-h', '"x-amz-server-side-encryption-customer-algorithm:AES256"', '-h', '"x-amz-server-side-encryption-customer-key:{}"'.format(TEST_ENCRYPTION_KEY_S3), '-h', '"x-amz-server-side-encryption-customer-key-md5:{}"'.format(TEST_ENCRYPTION_KEY_S3_MD5)]
    with SetBotoConfigForTest([('GSUtil', 'check_hashes', 'never')]):
        self.RunGsUtil(header_flags + ['cp', tmp_file, suri(bucket_uri1, 'test')])
        self.RunGsUtil(header_flags + ['rsync', suri(bucket_uri1), suri(bucket_uri2)])
        contents = self.RunGsUtil(header_flags + ['cat', suri(bucket_uri2, 'test')], return_stdout=True)
    self.assertEqual(contents, 'foo')