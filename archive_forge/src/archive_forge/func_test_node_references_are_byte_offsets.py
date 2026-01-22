from ... import errors, tests, transport
from .. import index as _mod_index
def test_node_references_are_byte_offsets(self):
    builder = _mod_index.GraphIndexBuilder(reference_lists=1)
    builder.add_node((b'reference',), b'data', ([],))
    builder.add_node((b'key',), b'data', ([(b'reference',)],))
    stream = builder.finish()
    contents = stream.read()
    self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=1\nkey_elements=1\nlen=2\nkey\x00\x0072\x00data\nreference\x00\x00\x00data\n\n', contents)