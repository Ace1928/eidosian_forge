import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
def test_parse_mappings_fails_for_duplicate_key(self):
    with testtools.ExpectedException(ValueError):
        self.parse(['key:val1', 'key:val2'])