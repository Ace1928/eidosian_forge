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
def test_rsync_to_nonexistent_bucket(self):
    """Tests that rsync from a non-existent bucket subdir fails gracefully."""
    tmpdir = self.CreateTempDir()
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1')
    self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.obj2')
    bucket_url_str = '%s://%s' % (self.default_provider, self.nonexistent_bucket_name)

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check():
        stderr = self.RunGsUtil(['rsync', '-d', bucket_url_str, tmpdir], expected_status=1, return_stderr=True)
        if self._use_gcloud_storage:
            self.assertIn('not found: 404', stderr)
        else:
            self.assertIn('Caught non-retryable exception', stderr)
        listing = TailSet(tmpdir, self.FlatListDir(tmpdir))
        self.assertEqual(listing, set(['/obj1', '/.obj2']))
    _Check()