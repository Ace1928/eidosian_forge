import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def make_index(self, size, recommended_pages=None):
    """Make an index with a generic size.

        This doesn't actually create anything on disk, it just primes a
        BTreeGraphIndex with the recommended information.
        """
    index = btree_index.BTreeGraphIndex(transport.get_transport_from_url('memory:///'), 'test-index', size=size)
    if recommended_pages is not None:
        index._recommended_pages = recommended_pages
    return index