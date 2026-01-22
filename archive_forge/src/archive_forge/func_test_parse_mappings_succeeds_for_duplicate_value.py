import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
def test_parse_mappings_succeeds_for_duplicate_value(self):
    self.assertEqual({'key1': 'val', 'key2': 'val'}, self.parse(['key1:val', 'key2:val'], False))