from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_iter_changes_empty_empty_is_empty(self):
    basis = self._get_map({}, maximum_size=10)
    target = self._get_map({}, chk_bytes=basis._store, maximum_size=10)
    self.assertEqual([], sorted(list(target.iter_changes(basis))))