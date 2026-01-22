import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
def test_empty_required_optional(self):
    self.parser.add_argument('--test-empty', metavar='req1=xxx,req2=yyy', action=parseractions.MultiKeyValueAction, dest='test_empty', default=None, required_keys=[], optional_keys=[], help='Test')
    results = self.parser.parse_args(['--test-empty', 'req1=aaa,req2=bbb', '--test-empty', 'req1=,req2='])
    actual = getattr(results, 'test_empty', [])
    expect = [{'req1': 'aaa', 'req2': 'bbb'}, {'req1': '', 'req2': ''}]
    self.assertCountEqual(expect, actual)