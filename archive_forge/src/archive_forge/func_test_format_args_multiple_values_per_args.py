import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
def test_format_args_multiple_values_per_args(self):
    li = utils.format_args(['K1=V1', 'K1=V2'])
    self.assertIn('K1', li)
    self.assertIn('V1', li['K1'])
    self.assertIn('V2', li['K1'])