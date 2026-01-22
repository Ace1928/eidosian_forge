import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
def test_mkvca_required(self):
    results = self.parser.parse_args(['--test', 'req1=aaa,bbb'])
    actual = getattr(results, 'test', [])
    expect = [{'req1': 'aaa,bbb'}]
    self.assertCountEqual(expect, actual)
    results = self.parser.parse_args(['--test', 'req1='])
    actual = getattr(results, 'test', [])
    expect = [{'req1': ''}]
    self.assertCountEqual(expect, actual)
    results = self.parser.parse_args(['--test', 'req1=aaa,bbb', '--test', 'req1='])
    actual = getattr(results, 'test', [])
    expect = [{'req1': 'aaa,bbb'}, {'req1': ''}]
    self.assertCountEqual(expect, actual)