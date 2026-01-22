from __future__ import absolute_import
import datetime
import re
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
@SkipForS3('Retention is not supported for s3 objects')
@SkipForXML('Retention is not supported for XML API')
def test_set_retention_unlocked_invalid_arg(self):
    bucket_uri = self.CreateBucket()
    stderr = self.RunGsUtil(['retention', 'set', '1a', suri(bucket_uri)], expected_status=1, return_stderr=True)
    self.assertRegex(stderr, 'Incorrect retention period specified')