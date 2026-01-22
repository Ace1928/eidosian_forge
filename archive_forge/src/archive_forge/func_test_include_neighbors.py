import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_include_neighbors(self):
    index = self.make_100_node_index()
    self.assertExpandOffsets([9, 10, 11, 12, 13, 14, 15], index, [12])
    self.assertExpandOffsets([88, 89, 90, 91, 92, 93, 94], index, [91])
    self.assertExpandOffsets([1, 2, 3, 4, 5, 6], index, [2])
    self.assertExpandOffsets([94, 95, 96, 97, 98, 99], index, [98])
    self.assertExpandOffsets([1, 2, 3, 80, 81, 82], index, [2, 81])
    self.assertExpandOffsets([1, 2, 3, 9, 10, 11, 80, 81, 82], index, [2, 10, 81])