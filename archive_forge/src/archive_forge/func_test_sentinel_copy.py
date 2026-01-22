import copy
from neutron_lib import constants
from neutron_lib.tests import _base as base
def test_sentinel_copy(self):
    singleton = constants.Sentinel()
    self.assertEqual(copy.deepcopy(singleton), copy.copy(singleton))