from tests.unit import AWSMockServiceTestCase
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.lifecycle import Rule, Lifecycle, Transition
def test_parse_transition_date(self):
    transition = self._get_bucket_lifecycle_config()[1].transition[0]
    self.assertEquals(transition.date, '2012-12-31T00:00:000Z')
    self.assertIsNone(transition.days)