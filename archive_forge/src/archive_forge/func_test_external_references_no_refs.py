import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_external_references_no_refs(self):
    index = self.make_index(ref_lists=0, nodes=[])
    self.assertRaises(ValueError, index.external_references, 0)