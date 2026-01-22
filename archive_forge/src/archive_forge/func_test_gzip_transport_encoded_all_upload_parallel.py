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
@SkipForS3('No compressed transport encoding support for S3.')
@SkipForXML('No compressed transport encoding support for the XML API.')
@SequentialAndParallelTransfer
def test_gzip_transport_encoded_all_upload_parallel(self):
    """Test gzip encoded files upload correctly."""
    file_names = ('test', 'test.txt', 'test.xml')
    local_uris = []
    bucket_uri = self.CreateBucket()
    tmpdir = self.CreateTempDir()
    contents = b'x' * 10000
    for file_name in file_names:
        local_uris.append(self.CreateTempFile(tmpdir, contents, file_name))
    stderr = self.RunGsUtil(['-D', '-m', 'rsync', '-J', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
    self.AssertNObjectsInBucket(bucket_uri, len(local_uris))
    for local_uri in local_uris:
        self.assertIn('Using compressed transport encoding for file://%s.' % local_uri, stderr)
    if not self._use_gcloud_storage:
        self.assertIn('send: Using gzip transport encoding for the request.', stderr)