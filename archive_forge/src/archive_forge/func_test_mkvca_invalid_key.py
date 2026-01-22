import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
def test_mkvca_invalid_key(self):
    try:
        self.parser.parse_args(['--test', 'req1=aaa,bbb='])
        self.fail('ArgumentTypeError should be raised')
    except argparse.ArgumentTypeError as e:
        self.assertIn('Invalid keys bbb specified.\nValid keys are:', str(e))
    try:
        self.parser.parse_args(['--test', 'nnn=aaa'])
        self.fail('ArgumentTypeError should be raised')
    except argparse.ArgumentTypeError as e:
        self.assertIn('Invalid keys nnn specified.\nValid keys are:', str(e))