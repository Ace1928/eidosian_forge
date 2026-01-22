import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
def test_format_args_semicolon(self):
    li = utils.format_args(['K1=V1;K2=V2;K3=V3;K4=V4;K5=V5'])
    self.assertEqual({'K1': 'V1', 'K2': 'V2', 'K3': 'V3', 'K4': 'V4', 'K5': 'V5'}, li)