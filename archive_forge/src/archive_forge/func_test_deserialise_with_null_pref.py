from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def test_deserialise_with_null_pref(self):
    node = self.module._deserialise_internal_node(b'chknode:\n10\n1\n1\npref\x00fo\n\x00\x00sha1:abcd\n', stuple(b'sha1:1234'))
    self.assertIsInstance(node, chk_map.InternalNode)
    self.assertEqual(1, len(node))
    self.assertEqual(10, node.maximum_size)
    self.assertEqual((b'sha1:1234',), node.key())
    self.assertEqual(b'pref\x00fo', node._search_prefix)
    self.assertEqual({b'pref\x00fo\x00': (b'sha1:abcd',)}, node._items)