from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def get_chk_bytes(self):
    if getattr(self, '_chk_bytes', None) is None:
        self._chk_bytes = super().get_chk_bytes()
    return self._chk_bytes