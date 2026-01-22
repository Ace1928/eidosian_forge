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
def test_move_bucket_to_dir(self):
    """Tests moving a local directory to a bucket."""
    bucket_uri = self.CreateBucket(test_objects=2)
    self.AssertNObjectsInBucket(bucket_uri, 2)
    tmpdir = self.CreateTempDir()
    self.RunGsUtil(['mv', suri(bucket_uri, '*'), tmpdir])
    dir_list = []
    for dirname, _, filenames in os.walk(tmpdir):
        for filename in filenames:
            dir_list.append(os.path.join(dirname, filename))
    self.assertEqual(len(dir_list), 2)
    self.AssertNObjectsInBucket(bucket_uri, 0)