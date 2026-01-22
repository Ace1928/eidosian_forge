from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
def test_rb_bucket_not_empty(self):
    bucket_uri = self.CreateBucket(test_objects=1)
    stderr = self.RunGsUtil(['rb', suri(bucket_uri)], expected_status=1, return_stderr=True)
    if self._use_gcloud_storage:
        self.assertIn('Bucket is not empty', stderr)
    else:
        self.assertIn('BucketNotEmpty', stderr)