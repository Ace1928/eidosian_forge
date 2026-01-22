import pytest
from srsly.msgpack import packb, unpackb, Packer, Unpacker, ExtType
from srsly.msgpack import PackOverflowError, PackValueError, UnpackValueError
def test_max_str_len():
    d = 'x' * 3
    packed = packb(d)
    unpacker = Unpacker(max_str_len=3, raw=False)
    unpacker.feed(packed)
    assert unpacker.unpack() == d
    unpacker = Unpacker(max_str_len=2, raw=False)
    with pytest.raises(UnpackValueError):
        unpacker.feed(packed)
        unpacker.unpack()