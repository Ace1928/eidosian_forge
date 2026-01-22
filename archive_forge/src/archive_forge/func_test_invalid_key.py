import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
def test_invalid_key(self):
    self.assertRaises(argparse.ArgumentTypeError, self.parser.parse_args, ['--test', 'req1=aaa,req2=bbb,aaa=req1'])