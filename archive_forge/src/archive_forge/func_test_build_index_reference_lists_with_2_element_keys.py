from ... import errors, tests, transport
from .. import index as _mod_index
def test_build_index_reference_lists_with_2_element_keys(self):
    builder = _mod_index.GraphIndexBuilder(reference_lists=1, key_elements=2)
    builder.add_node((b'key', b'key2'), b'data', ([],))
    stream = builder.finish()
    contents = stream.read()
    self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=1\nkey_elements=2\nlen=1\nkey\x00key2\x00\x00\x00data\n\n', contents)