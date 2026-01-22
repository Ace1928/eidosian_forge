from array import array
from srsly.msgpack import packb, unpackb
def test_str16_from_float():
    _runtest('f', 2 ** 8, b'\xda', b'\x01\x00', False)
    _runtest('f', 2 ** 16 - 4, b'\xda', b'\xff\xfc', False)