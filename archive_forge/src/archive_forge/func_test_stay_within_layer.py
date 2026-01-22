import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_stay_within_layer(self):
    index = self.make_1000_node_index()
    self.assertExpandOffsets([1, 2, 3, 4], index, [2])
    self.assertExpandOffsets([6, 7, 8, 9], index, [6])
    self.assertExpandOffsets([6, 7, 8, 9], index, [9])
    self.assertExpandOffsets([10, 11, 12, 13, 14, 15], index, [10])
    self.assertExpandOffsets([10, 11, 12, 13, 14, 15, 16], index, [13])
    self.set_cached_offsets(index, [0, 4, 12])
    self.assertExpandOffsets([5, 6, 7, 8, 9], index, [7])
    self.assertExpandOffsets([10, 11], index, [11])