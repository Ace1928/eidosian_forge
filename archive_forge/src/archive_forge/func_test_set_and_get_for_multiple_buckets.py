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
def test_set_and_get_for_multiple_buckets(self):
    bucket1_uri = self.CreateBucket()
    bucket2_uri = self.CreateBucket()
    new_storage_class = 'nearline'
    stderr = self.RunGsUtil(self._set_dsc_cmd + [new_storage_class, suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
    for bucket_uri in (suri(bucket1_uri), suri(bucket2_uri)):
        if not self._use_gcloud_storage:
            self.assertRegexpMatchesWithFlags(stderr, 'Setting default storage class to "%s" for bucket %s' % (new_storage_class, bucket_uri), flags=re.IGNORECASE)
    stdout = self.RunGsUtil(self._get_dsc_cmd + [suri(bucket1_uri), suri(bucket2_uri)], return_stdout=True)
    for bucket_uri in (suri(bucket1_uri), suri(bucket2_uri)):
        self.assertRegexpMatchesWithFlags(stdout, '%s:\\s+%s' % (bucket_uri, new_storage_class), flags=re.IGNORECASE)