import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_external_references_multiple_ref_lists(self):
    missing_key = (b'missing',)
    index = self.make_index(ref_lists=2, nodes=[((b'key',), b'value', ([], [missing_key]))])
    self.assertEqual(set(), index.external_references(0))
    self.assertEqual({missing_key}, index.external_references(1))