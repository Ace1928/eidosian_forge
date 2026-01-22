import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
def test_error_values(self):
    data_list = [['--hint', 'red'], ['--hint', '='], ['--hint', '=red']]
    for data in data_list:
        self.assertRaises(argparse.ArgumentTypeError, self.parser.parse_args, data)