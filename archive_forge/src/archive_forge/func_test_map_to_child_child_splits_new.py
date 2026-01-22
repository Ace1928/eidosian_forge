from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_map_to_child_child_splits_new(self):
    chkmap = self._get_map({(b'k1',): b'foo', (b'k22',): b'bar'}, maximum_size=10)
    self.assertEqualDiff("'' InternalNode\n  'k1' LeafNode\n      ('k1',) 'foo'\n  'k2' LeafNode\n      ('k22',) 'bar'\n", chkmap._dump_tree())
    chkmap = CHKMap(chkmap._store, chkmap._root_node)
    chkmap._ensure_root()
    node = chkmap._root_node
    self.assertEqual(2, len([value for value in node._items.values() if isinstance(value, StaticTuple)]))
    prefix, nodes = node.map(chkmap._store, (b'k23',), b'quux')
    self.assertEqual(b'k', prefix)
    self.assertEqual([(b'', node)], nodes)
    child = node._items[b'k2']
    self.assertIsInstance(child, InternalNode)
    self.assertEqual(2, len(child))
    self.assertEqual({(b'k22',): b'bar', (b'k23',): b'quux'}, self.to_dict(child, None))
    self.assertEqual(None, child._key)
    self.assertEqual(10, child.maximum_size)
    self.assertEqual(1, child._key_width)
    self.assertEqual(3, child._node_width)
    self.assertEqual(3, len(chkmap))
    self.assertEqual({(b'k1',): b'foo', (b'k22',): b'bar', (b'k23',): b'quux'}, self.to_dict(chkmap))
    keys = list(node.serialise(chkmap._store))
    child_key = child._key
    k22_key = child._items[b'k22']._key
    k23_key = child._items[b'k23']._key
    self.assertEqual({k22_key, k23_key, child_key, node.key()}, set(keys))
    self.assertEqualDiff("'' InternalNode\n  'k1' LeafNode\n      ('k1',) 'foo'\n  'k2' InternalNode\n    'k22' LeafNode\n      ('k22',) 'bar'\n    'k23' LeafNode\n      ('k23',) 'quux'\n", chkmap._dump_tree())