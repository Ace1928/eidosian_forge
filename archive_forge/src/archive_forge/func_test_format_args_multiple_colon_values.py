import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
def test_format_args_multiple_colon_values(self):
    li = utils.format_args(['K1=V1', 'K2=V2,V22,V222,V2222', 'K3=3.3.3.3'])
    self.assertEqual({'K1': 'V1', 'K2': 'V2,V22,V222,V2222', 'K3': '3.3.3.3'}, li)