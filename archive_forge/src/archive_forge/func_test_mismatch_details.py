import testtools
from testtools import matchers as tt_matchers
from keystoneauth1.tests.unit import matchers as ks_matchers
def test_mismatch_details(self):
    examples = self.describe_examples
    for difference, matchee, matcher in examples:
        mismatch = matcher.match(matchee)
        details = mismatch.get_details()
        self.assertEqual(dict(details), details)