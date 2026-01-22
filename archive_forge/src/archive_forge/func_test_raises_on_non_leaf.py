from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def test_raises_on_non_leaf(self):
    self.assertDeserialiseErrors(b'')
    self.assertDeserialiseErrors(b'short\n')
    self.assertDeserialiseErrors(b'chknotleaf:\n')
    self.assertDeserialiseErrors(b'chkleaf:x\n')
    self.assertDeserialiseErrors(b'chkleaf:\n')
    self.assertDeserialiseErrors(b'chkleaf:\nnotint\n')
    self.assertDeserialiseErrors(b'chkleaf:\n10\n')
    self.assertDeserialiseErrors(b'chkleaf:\n10\n256\n')
    self.assertDeserialiseErrors(b'chkleaf:\n10\n256\n10\n')