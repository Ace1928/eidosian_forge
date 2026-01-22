from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def test_iteritems_selected_one_of_two_items(self):
    node = self.module._deserialise_leaf_node(b'chkleaf:\n0\n1\n2\n\nfoo bar\x001\nbaz\nquux\x001\nblarh\n', (b'sha1:1234',))
    self.assertEqual(2, len(node))
    self.assertEqual([((b'quux',), b'blarh')], sorted(node.iteritems(None, [(b'quux',), (b'qaz',)])))