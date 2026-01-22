from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from gslib.cs_api_map import ApiSelector
from gslib.tests.test_cp import TestCpMvPOSIXBucketToLocalErrors
from gslib.tests.test_cp import TestCpMvPOSIXBucketToLocalNoErrors
from gslib.tests.test_cp import TestCpMvPOSIXLocalToBucketNoErrors
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SequentialAndParallelTransfer
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils import shim_util
@unittest.skipIf(IS_WINDOWS, 'POSIX attributes not available on Windows.')
def test_mv_preserve_posix_bucket_to_dir_errors(self):
    """Tests use of the -P flag with mv from a bucket to a local dir.

    Specifically, combinations of POSIX attributes in metadata that will fail
    validation.
    """
    bucket_uri = self.CreateBucket()
    tmpdir = self.CreateTempDir()
    obj = self.CreateObject(bucket_uri=bucket_uri, object_name='obj', contents=b'obj')
    TestCpMvPOSIXBucketToLocalErrors(self, bucket_uri, obj, tmpdir, is_cp=False)