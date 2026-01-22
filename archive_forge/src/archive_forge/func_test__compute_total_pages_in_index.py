import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test__compute_total_pages_in_index(self):
    index = self.make_index(None)
    self.assertNumPages(1, index, 1024)
    self.assertNumPages(1, index, 4095)
    self.assertNumPages(1, index, 4096)
    self.assertNumPages(2, index, 4097)
    self.assertNumPages(2, index, 8192)
    self.assertNumPages(76, index, 4096 * 75 + 10)