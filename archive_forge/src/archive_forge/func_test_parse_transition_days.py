from tests.unit import AWSMockServiceTestCase
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.lifecycle import Rule, Lifecycle, Transition
def test_parse_transition_days(self):
    transition = self._get_bucket_lifecycle_config()[0].transition[0]
    self.assertEquals(transition.days, 30)
    self.assertIsNone(transition.date)