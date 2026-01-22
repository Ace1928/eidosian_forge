from tests.unit import AWSMockServiceTestCase
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.lifecycle import Rule, Lifecycle, Transition
def test_expiration_and_transition(self):
    t = Transition(date='2012-11-30T00:00:000Z', storage_class='GLACIER')
    r = Rule('myid', 'prefix', 'Enabled', expiration=30, transition=t)
    xml = r.to_xml()
    self.assertIn('<Transition><StorageClass>GLACIER</StorageClass><Date>2012-11-30T00:00:000Z</Date>', xml)
    self.assertIn('<Expiration><Days>30</Days></Expiration>', xml)