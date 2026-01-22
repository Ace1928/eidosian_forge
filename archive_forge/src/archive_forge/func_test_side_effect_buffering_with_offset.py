from ... import errors, tests, transport
from .. import index as _mod_index
def test_side_effect_buffering_with_offset(self):
    nodes = self.make_nodes(20)
    index = self.make_index_with_offset(offset=1234567, nodes=nodes)
    index._transport.recommended_page_size = lambda: 64 * 1024
    subset_nodes = [nodes[0][0], nodes[10][0], nodes[19][0]]
    entries = [n[1] for n in index.iter_entries(subset_nodes)]
    self.assertEqual(sorted(subset_nodes), sorted(entries))
    self.assertEqual(20, index.key_count())