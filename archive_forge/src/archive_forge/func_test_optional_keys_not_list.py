import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
def test_optional_keys_not_list(self):
    self.assertRaises(TypeError, self.parser.add_argument, '--test-optional-dict', metavar='req1=xxx,req2=yyy', action=parseractions.MultiKeyValueAction, dest='test_optional_dict', default=None, required_keys=['req1', 'req2'], optional_keys={'aaa': 'bbb'}, help='Test')