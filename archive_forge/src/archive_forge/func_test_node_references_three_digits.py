from ... import errors, tests, transport
from .. import index as _mod_index
def test_node_references_three_digits(self):
    builder = _mod_index.GraphIndexBuilder(reference_lists=1)
    references = [(b'%d' % val,) for val in range(8, -1, -1)]
    builder.add_node((b'2-key',), b'', (references,))
    stream = builder.finish()
    contents = stream.read()
    self.assertEqualDiff(b'Bazaar Graph Index 1\nnode_ref_lists=1\nkey_elements=1\nlen=1\n0\x00a\x00\x00\n1\x00a\x00\x00\n2\x00a\x00\x00\n2-key\x00\x00151\r145\r139\r133\r127\r121\r071\r065\r059\x00\n3\x00a\x00\x00\n4\x00a\x00\x00\n5\x00a\x00\x00\n6\x00a\x00\x00\n7\x00a\x00\x00\n8\x00a\x00\x00\n\n', contents)