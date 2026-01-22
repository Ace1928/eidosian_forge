from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
def test_rb_minus_f(self):
    bucket_uri = self.CreateBucket()
    stderr = self.RunGsUtil(['rb', '-f', 'gs://%s' % self.nonexistent_bucket_name, suri(bucket_uri)], return_stderr=True, expected_status=1)
    self.assertNotIn('bucket does not exist.', stderr)
    stderr = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stderr=True, expected_status=1)
    self.assertIn('404', stderr)