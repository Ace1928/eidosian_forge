from ... import errors, tests, transport
from .. import index as _mod_index
def test_build_index_one_node_2_element_keys(self):
    builder = _mod_index.GraphIndexBuilder(key_elements=2)
    builder.add_node((b'akey', b'secondpart'), b'data')
    stream = builder.finish()
    contents = stream.read()
    self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=0\nkey_elements=2\nlen=1\nakey\x00secondpart\x00\x00\x00data\n\n', contents)