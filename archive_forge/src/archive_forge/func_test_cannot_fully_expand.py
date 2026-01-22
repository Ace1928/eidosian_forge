import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_cannot_fully_expand(self):
    index = self.make_100_node_index()
    self.set_cached_offsets(index, [0, 10, 12])
    self.assertExpandOffsets([11], index, [11])