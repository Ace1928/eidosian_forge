from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def test_key_after_unmap(self):
    node = self.module._deserialise_leaf_node(b'chkleaf:\n0\n1\n2\n\nfoo bar\x001\nbaz\nquux\x001\nblarh\n', (b'sha1:1234',))
    node.unmap(None, (b'foo bar',))
    self.assertEqual(None, node.key())