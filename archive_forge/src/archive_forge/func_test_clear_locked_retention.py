from __future__ import absolute_import
import datetime
import re
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
@SkipForS3('Retention is not supported for s3 objects')
@SkipForXML('Retention is not supported for XML API')
def test_clear_locked_retention(self):
    bucket_uri = self.CreateBucketWithRetentionPolicy(retention_period_in_seconds=_SECONDS_IN_DAY, is_locked=True)
    stderr = self.RunGsUtil(['retention', 'clear', suri(bucket_uri)], expected_status=1, return_stderr=True)
    self.assertRegex(stderr, 'Bucket .* has a locked Retention Policy which cannot be removed')