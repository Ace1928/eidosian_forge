from tests.unit import AWSMockServiceTestCase
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.lifecycle import Rule, Lifecycle, Transition
def test_parse_storage_class_standard_ia(self):
    transition = self._get_bucket_lifecycle_config()[1].transition[0]
    self.assertEqual(transition.storage_class, 'STANDARD_IA')