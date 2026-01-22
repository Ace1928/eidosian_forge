import tempfile
import unittest
from boto.compat import StringIO, six, json
from textwrap import dedent
from boto.cloudfront.distribution import Distribution
def test_create_canned_policy(self):
    """
        Test that a canned policy is generated correctly.
        """
    url = 'http://1234567.cloudfront.com/test_resource.mp3?dog=true'
    expires = 999999
    policy = self.dist._canned_policy(url, expires)
    policy = json.loads(policy)
    self.assertEqual(1, len(policy.keys()))
    statements = policy['Statement']
    self.assertEqual(1, len(statements))
    statement = statements[0]
    resource = statement['Resource']
    self.assertEqual(url, resource)
    condition = statement['Condition']
    self.assertEqual(1, len(condition.keys()))
    date_less_than = condition['DateLessThan']
    self.assertEqual(1, len(date_less_than.keys()))
    aws_epoch_time = date_less_than['AWS:EpochTime']
    self.assertEqual(expires, aws_epoch_time)