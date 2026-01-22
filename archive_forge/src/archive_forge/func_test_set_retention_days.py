from __future__ import absolute_import
import datetime
import re
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
@SkipForS3('Retention is not supported for s3 objects')
@SkipForXML('Retention is not supported for XML API')
def test_set_retention_days(self):
    bucket_uri = self.CreateBucket()
    self.RunGsUtil(['retention', 'set', '1d', suri(bucket_uri)])
    self.VerifyRetentionPolicy(bucket_uri, expected_retention_period_in_seconds=_SECONDS_IN_DAY)