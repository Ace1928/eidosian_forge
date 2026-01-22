from ... import errors, tests, transport
from .. import index as _mod_index
def test_add_node_referencing_missing_key_makes_absent(self):
    builder = _mod_index.GraphIndexBuilder(reference_lists=1)
    builder.add_node((b'rey',), b'data', ([(b'beference',), (b'aeference2',)],))
    stream = builder.finish()
    contents = stream.read()
    self.assertEqual(b'Bazaar Graph Index 1\nnode_ref_lists=1\nkey_elements=1\nlen=1\naeference2\x00a\x00\x00\nbeference\x00a\x00\x00\nrey\x00\x00074\r059\x00data\n\n', contents)