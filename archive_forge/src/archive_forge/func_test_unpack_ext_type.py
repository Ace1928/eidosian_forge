import array
from srsly import msgpack
from srsly.msgpack._ext_type import ExtType
def test_unpack_ext_type():

    def check(b, expected):
        assert msgpack.unpackb(b) == expected
    check(b'\xd4BA', ExtType(66, b'A'))
    check(b'\xd5BAB', ExtType(66, b'AB'))
    check(b'\xd6BABCD', ExtType(66, b'ABCD'))
    check(b'\xd7BABCDEFGH', ExtType(66, b'ABCDEFGH'))
    check(b'\xd8B' + b'A' * 16, ExtType(66, b'A' * 16))
    check(b'\xc7\x03BABC', ExtType(66, b'ABC'))
    check(b'\xc8\x01#B' + b'A' * 291, ExtType(66, b'A' * 291))
    check(b'\xc9\x00\x01#EB' + b'A' * 74565, ExtType(66, b'A' * 74565))