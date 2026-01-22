from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_iter_changes_ab_ab_nodes_not_loaded(self):
    basis = self._get_map({(b'a',): b'content here', (b'b',): b'more content'}, maximum_size=10)
    target = self._get_map({(b'a',): b'content here', (b'b',): b'more content'}, chk_bytes=basis._store, maximum_size=10)
    list(target.iter_changes(basis))
    self.assertIsInstance(target._root_node, StaticTuple)
    self.assertIsInstance(basis._root_node, StaticTuple)