import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
def test_no_nets(self):
    nets = []
    result = utils.parse_nets(nets)
    self.assertEqual([], result)