from pytest import raises
import datetime
from srsly.msgpack import packb, unpackb, Unpacker, FormatError, StackError, OutOfData
def test_strict_map_key():
    valid = {u'unicode': 1, b'bytes': 2}
    packed = packb(valid, use_bin_type=True)
    assert valid == unpackb(packed, raw=False, strict_map_key=True)
    invalid = {42: 1}
    packed = packb(invalid, use_bin_type=True)
    with raises(ValueError):
        unpackb(packed, raw=False, strict_map_key=True)