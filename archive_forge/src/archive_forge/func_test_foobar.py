import io
import pytest
from srsly.msgpack import Unpacker, BufferFull
from srsly.msgpack import pack
from srsly.msgpack.exceptions import OutOfData
def test_foobar():
    unpacker = Unpacker(read_size=3, use_list=1)
    unpacker.feed(b'foobar')
    assert unpacker.unpack() == ord(b'f')
    assert unpacker.unpack() == ord(b'o')
    assert unpacker.unpack() == ord(b'o')
    assert unpacker.unpack() == ord(b'b')
    assert unpacker.unpack() == ord(b'a')
    assert unpacker.unpack() == ord(b'r')
    with pytest.raises(OutOfData):
        unpacker.unpack()
    unpacker.feed(b'foo')
    unpacker.feed(b'bar')
    k = 0
    for o, e in zip(unpacker, 'foobarbaz'):
        assert o == ord(e)
        k += 1
    assert k == len(b'foobar')