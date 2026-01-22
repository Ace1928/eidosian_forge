from ... import errors, tests, transport
from .. import index as _mod_index
def test_add_node_bad_key_in_reference_lists(self):
    builder = _mod_index.GraphIndexBuilder(reference_lists=1)
    self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'akey',), b'data aa', ([(b'a key',)],))
    self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'akey',), b'data aa', (['not-a-tuple'],))
    self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'akey',), b'data aa', ([()],))
    self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'akey',), b'data aa', ([(b'primary', b'secondary')],))
    self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'akey',), b'data aa', ([(b'agoodkey',), (b'that is a bad key',)],))
    builder = _mod_index.GraphIndexBuilder(reference_lists=2)
    self.assertRaises(_mod_index.BadIndexKey, builder.add_node, (b'akey',), b'data aa', ([], ['a bad key']))