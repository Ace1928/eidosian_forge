import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_unknown_size(self):
    index = self.make_index(None, 10)
    self.assertExpandOffsets([0], index, [0])
    self.assertExpandOffsets([1, 4, 9], index, [1, 4, 9])