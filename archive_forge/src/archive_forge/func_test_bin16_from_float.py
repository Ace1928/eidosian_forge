from array import array
from srsly.msgpack import packb, unpackb
def test_bin16_from_float():
    _runtest('f', 2 ** 8, b'\xc5', b'\x01\x00', True)
    _runtest('f', 2 ** 16 - 4, b'\xc5', b'\xff\xfc', True)