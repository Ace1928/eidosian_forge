import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
def test_zero_values(self):
    results = self.parser.parse_args('--foo 0'.split())
    actual = getattr(results, 'foo', None)
    self.assertEqual(actual, 0)