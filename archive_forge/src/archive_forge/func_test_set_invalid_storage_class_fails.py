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
def test_set_invalid_storage_class_fails(self):
    bucket_uri = self.CreateBucket()
    stderr = self.RunGsUtil(self._set_dsc_cmd + ['invalidclass', suri(bucket_uri)], return_stderr=True, expected_status=1)
    if self._use_gcloud_storage:
        self.assertIn('Invalid storage class', stderr)
    else:
        self.assertIn('BadRequestException: 400', stderr)