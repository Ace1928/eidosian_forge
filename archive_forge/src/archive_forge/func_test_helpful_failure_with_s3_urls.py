from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
from unittest import skipIf
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
def test_helpful_failure_with_s3_urls(self):
    s3_bucket_url = 's3://somebucket'
    failure_msg = 'does not support the URL "%s"' % s3_bucket_url
    stderr = self.RunGsUtil(self._get_dsc_cmd + [s3_bucket_url], return_stderr=True, expected_status=1)
    self.assertIn(failure_msg, stderr)
    stderr = self.RunGsUtil(self._set_dsc_cmd + ['ClassFoo', s3_bucket_url], return_stderr=True, expected_status=1)
    self.assertIn(failure_msg, stderr)