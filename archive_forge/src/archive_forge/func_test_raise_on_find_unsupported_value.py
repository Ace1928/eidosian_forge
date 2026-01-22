from pytest import raises
import datetime
from srsly.msgpack import packb, unpackb, Unpacker, FormatError, StackError, OutOfData
def test_raise_on_find_unsupported_value():
    with raises(TypeError):
        packb(datetime.datetime.now())