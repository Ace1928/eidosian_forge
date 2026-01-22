from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_iteritems_keys_prefixed_by_2_width_one_leaf(self):
    chkmap = self._get_map({(b'a', b'a'): b'content here', (b'a', b'b'): b'more content', (b'b', b''): b'boring content'}, key_width=2)
    self.assertEqual({(b'a', b'a'): b'content here', (b'a', b'b'): b'more content'}, self.to_dict(chkmap, [(b'a',)]))