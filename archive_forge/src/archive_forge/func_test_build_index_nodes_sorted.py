from ... import errors, tests, transport
from .. import index as _mod_index
def test_build_index_nodes_sorted(self):
    builder = _mod_index.GraphIndexBuilder()
    builder.add_node((b'2002',), b'data')
    builder.add_node((b'2000',), b'data')
    builder.add_node((b'2001',), b'data')
    stream = builder.finish()
    contents = stream.read()
    self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=0\nkey_elements=1\nlen=3\n2000\x00\x00\x00data\n2001\x00\x00\x00data\n2002\x00\x00\x00data\n\n', contents)