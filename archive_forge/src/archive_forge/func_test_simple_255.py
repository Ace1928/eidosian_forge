from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def test_simple_255(self):
    self.assertSearchKey255(b'\x8cse!', stuple(b'foo'))
    self.assertSearchKey255(b'\x8cse!\x00\x8cse!', stuple(b'foo', b'foo'))
    self.assertSearchKey255(b'\x8cse!\x00v\xff\x8c\xaa', stuple(b'foo', b'bar'))
    self.assertSearchKey255(b'\xfdm\x93_\x00P_\x1bL', stuple(b'<', b'V'))