from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_iter_changes_empty_ab(self):
    basis = self._get_map({}, maximum_size=10)
    target = self._get_map({(b'a',): b'content here', (b'b',): b'more content'}, chk_bytes=basis._store, maximum_size=10)
    self.assertEqual([((b'a',), None, b'content here'), ((b'b',), None, b'more content')], sorted(list(target.iter_changes(basis))))