from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_common_prefix(self):
    self.assertCommonPrefix(b'beg', b'beg', b'begin')