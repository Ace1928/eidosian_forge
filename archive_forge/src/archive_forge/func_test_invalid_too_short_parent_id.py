from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def test_invalid_too_short_parent_id(self):
    self.assertBytesToTextKeyRaises(b'file:file-id\nparent-id')