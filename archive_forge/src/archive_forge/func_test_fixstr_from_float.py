from array import array
from srsly.msgpack import packb, unpackb
def test_fixstr_from_float():
    _runtest('f', 4, b'\xa4', b'', False)
    _runtest('f', 28, b'\xbc', b'', False)