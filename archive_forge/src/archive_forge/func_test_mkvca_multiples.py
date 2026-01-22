import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
def test_mkvca_multiples(self):
    results = self.parser.parse_args(['--test', 'req1=aaa,bbb,opt2=ccc'])
    actual = getattr(results, 'test', [])
    expect = [{'req1': 'aaa,bbb', 'opt2': 'ccc'}]
    self.assertCountEqual(expect, actual)