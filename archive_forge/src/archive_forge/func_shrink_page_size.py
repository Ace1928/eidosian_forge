import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def shrink_page_size(self):
    """Shrink the default page size so that less fits in a page."""
    self.overrideAttr(btree_index, '_PAGE_SIZE')
    btree_index._PAGE_SIZE = 2048