import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_small_requests_unexpanded(self):
    index = self.make_100_node_index()
    self.set_cached_offsets(index, [0])
    self.assertExpandOffsets([1], index, [1])
    self.assertExpandOffsets([50], index, [50])
    self.assertExpandOffsets([49, 50, 51, 59, 60, 61], index, [50, 60])
    index = self.make_1000_node_index()
    self.set_cached_offsets(index, [0])
    self.assertExpandOffsets([1], index, [1])
    self.set_cached_offsets(index, [0, 1])
    self.assertExpandOffsets([100], index, [100])
    self.set_cached_offsets(index, [0, 1, 100])
    self.assertExpandOffsets([2, 3, 4, 5, 6, 7], index, [2])
    self.assertExpandOffsets([2, 3, 4, 5, 6, 7], index, [4])
    self.set_cached_offsets(index, [0, 1, 2, 3, 4, 5, 6, 7, 100])
    self.assertExpandOffsets([102, 103, 104, 105, 106, 107, 108], index, [105])