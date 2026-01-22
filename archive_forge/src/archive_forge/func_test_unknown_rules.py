import sys
from breezy import rules, tests
def test_unknown_rules(self):
    err = rules.UnknownRules(['foo', 'bar'])
    self.assertEqual('Unknown rules detected: foo, bar.', str(err))