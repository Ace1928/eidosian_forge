import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
def test_mkvca_no_required_optional(self):
    self.parser.add_argument('--test-empty', metavar='req1=xxx,yyy', action=parseractions.MultiKeyValueCommaAction, dest='test_empty', default=None, required_keys=[], optional_keys=[], help='Test')
    results = self.parser.parse_args(['--test-empty', 'req1=aaa,bbb'])
    actual = getattr(results, 'test_empty', [])
    expect = [{'req1': 'aaa,bbb'}]
    self.assertCountEqual(expect, actual)
    results = self.parser.parse_args(['--test-empty', 'xyz=aaa,bbb'])
    actual = getattr(results, 'test_empty', [])
    expect = [{'xyz': 'aaa,bbb'}]
    self.assertCountEqual(expect, actual)