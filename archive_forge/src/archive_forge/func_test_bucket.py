from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.constants import UTF8
from gslib.utils.retry_util import Retry
def test_bucket(self):
    bucket_uri = self.CreateBucket()
    obj_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'foo')

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check():
        stdout = self.RunGsUtil(['du', suri(bucket_uri)], return_stdout=True)
        self.assertEqual(stdout, '%-11s  %s\n' % (3, suri(obj_uri)))
    _Check()