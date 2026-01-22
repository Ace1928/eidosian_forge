from ... import errors, tests, transport
from .. import index as _mod_index
def test_build_index_one_node_no_refs(self):
    builder = _mod_index.GraphIndexBuilder()
    builder.add_node((b'akey',), b'data')
    stream = builder.finish()
    contents = stream.read()
    self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=0\nkey_elements=1\nlen=1\nakey\x00\x00\x00data\n\n', contents)