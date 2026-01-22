from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def test_simple_16(self):
    self.assertSearchKey16(b'8C736521', stuple(b'foo'))
    self.assertSearchKey16(b'8C736521\x008C736521', stuple(b'foo', b'foo'))
    self.assertSearchKey16(b'8C736521\x0076FF8CAA', stuple(b'foo', b'bar'))
    self.assertSearchKey16(b'ED82CD11', stuple(b'abcd'))