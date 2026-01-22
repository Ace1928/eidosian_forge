from array import array
from srsly.msgpack import packb, unpackb
def test_bin16_from_byte():
    _runtest('B', 2 ** 8, b'\xc5', b'\x01\x00', True)
    _runtest('B', 2 ** 16 - 1, b'\xc5', b'\xff\xff', True)