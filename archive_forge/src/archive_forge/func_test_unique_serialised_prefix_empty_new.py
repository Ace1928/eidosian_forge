from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_unique_serialised_prefix_empty_new(self):
    node = LeafNode()
    self.assertIs(None, node._compute_search_prefix())