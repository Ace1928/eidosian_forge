import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
def test_positive_values(self):
    results = self.parser.parse_args('--foo 1'.split())
    actual = getattr(results, 'foo', None)
    self.assertEqual(actual, 1)