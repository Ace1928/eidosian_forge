import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
def test_parse_mappings_succeeds_for_one_mapping(self):
    self.assertEqual({'key': 'val'}, self.parse(['key:val']))