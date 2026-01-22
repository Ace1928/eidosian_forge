from pytest import raises
import datetime
from srsly.msgpack import packb, unpackb, Unpacker, FormatError, StackError, OutOfData
def test_invalidvalue():
    incomplete = b'\xd9\x97#DL_'
    with raises(ValueError):
        unpackb(incomplete)
    with raises(OutOfData):
        unpacker = Unpacker()
        unpacker.feed(incomplete)
        unpacker.unpack()
    with raises(FormatError):
        unpackb(b'\xc1')
    with raises(FormatError):
        unpackb(b'\x91\xc1')
    with raises(StackError):
        unpackb(b'\x91' * 3000)