import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
def test_parse_mappings_succeeds_for_no_mappings(self):
    self.assertEqual({}, self.parse(['']))