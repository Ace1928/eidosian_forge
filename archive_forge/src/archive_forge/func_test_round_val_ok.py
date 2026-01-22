import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
def test_round_val_ok(self):
    for expected, value in ((0, 0), (0, 0.1), (1, 0.5), (1, 1.49), (2, 1.5)):
        self.assertEqual(expected, helpers.round_val(value))