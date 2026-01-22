import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def make_index_with_offset(self, ref_lists=1, key_elements=1, nodes=[], offset=0):
    builder = btree_index.BTreeBuilder(key_elements=key_elements, reference_lists=ref_lists)
    builder.add_nodes(nodes)
    transport = self.get_transport('')
    temp_file = builder.finish()
    content = temp_file.read()
    del temp_file
    size = len(content)
    transport.put_bytes('index', b' ' * offset + content)
    return btree_index.BTreeGraphIndex(transport, 'index', size=size, offset=offset)