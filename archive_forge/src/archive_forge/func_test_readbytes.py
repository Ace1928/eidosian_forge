import io
import pytest
from srsly.msgpack import Unpacker, BufferFull
from srsly.msgpack import pack
from srsly.msgpack.exceptions import OutOfData
def test_readbytes():
    unpacker = Unpacker(read_size=3)
    unpacker.feed(b'foobar')
    assert unpacker.unpack() == ord(b'f')
    assert unpacker.read_bytes(3) == b'oob'
    assert unpacker.unpack() == ord(b'a')
    assert unpacker.unpack() == ord(b'r')
    unpacker = Unpacker(io.BytesIO(b'foobar'), read_size=3)
    assert unpacker.unpack() == ord(b'f')
    assert unpacker.read_bytes(3) == b'oob'
    assert unpacker.unpack() == ord(b'a')
    assert unpacker.unpack() == ord(b'r')