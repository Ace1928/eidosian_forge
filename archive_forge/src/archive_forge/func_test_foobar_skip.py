import io
import pytest
from srsly.msgpack import Unpacker, BufferFull
from srsly.msgpack import pack
from srsly.msgpack.exceptions import OutOfData
def test_foobar_skip():
    unpacker = Unpacker(read_size=3, use_list=1)
    unpacker.feed(b'foobar')
    assert unpacker.unpack() == ord(b'f')
    unpacker.skip()
    assert unpacker.unpack() == ord(b'o')
    unpacker.skip()
    assert unpacker.unpack() == ord(b'a')
    unpacker.skip()
    with pytest.raises(OutOfData):
        unpacker.unpack()