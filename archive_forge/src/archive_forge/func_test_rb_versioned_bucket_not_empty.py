from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
def test_rb_versioned_bucket_not_empty(self):
    bucket_uri = self.CreateVersionedBucket(test_objects=1)
    stderr = self.RunGsUtil(['rb', suri(bucket_uri)], expected_status=1, return_stderr=True)
    self.assertIn('Bucket is not empty. Note: this is a versioned bucket', stderr)