import tempfile
import unittest
from boto.compat import StringIO, six, json
from textwrap import dedent
from boto.cloudfront.distribution import Distribution
def test_custom_policy_valid_after(self):
    """
        Test that a custom policy can be created with a valid-after time and
        an arbitrary URL.
        """
    url = 'http://1234567.cloudfront.com/*'
    valid_after = 999999
    policy = self.dist._custom_policy(url, valid_after=valid_after)
    policy = json.loads(policy)
    self.assertEqual(1, len(policy.keys()))
    statements = policy['Statement']
    self.assertEqual(1, len(statements))
    statement = statements[0]
    resource = statement['Resource']
    self.assertEqual(url, resource)
    condition = statement['Condition']
    self.assertEqual(2, len(condition.keys()))
    date_less_than = condition['DateLessThan']
    date_greater_than = condition['DateGreaterThan']
    self.assertEqual(1, len(date_greater_than.keys()))
    aws_epoch_time = date_greater_than['AWS:EpochTime']
    self.assertEqual(valid_after, aws_epoch_time)