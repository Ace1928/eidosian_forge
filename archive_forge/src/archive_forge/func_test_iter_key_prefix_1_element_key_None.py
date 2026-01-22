import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_iter_key_prefix_1_element_key_None(self):
    index = self.make_index()
    self.assertRaises(_mod_index.BadIndexKey, list, index.iter_entries_prefix([(None,)]))