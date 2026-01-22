import array
from srsly import msgpack
from srsly.msgpack._ext_type import ExtType
def test_pack_ext_type():

    def p(s):
        packer = msgpack.Packer()
        packer.pack_ext_type(66, s)
        return packer.bytes()
    assert p(b'A') == b'\xd4BA'
    assert p(b'AB') == b'\xd5BAB'
    assert p(b'ABCD') == b'\xd6BABCD'
    assert p(b'ABCDEFGH') == b'\xd7BABCDEFGH'
    assert p(b'A' * 16) == b'\xd8B' + b'A' * 16
    assert p(b'ABC') == b'\xc7\x03BABC'
    assert p(b'A' * 291) == b'\xc8\x01#B' + b'A' * 291
    assert p(b'A' * 74565) == b'\xc9\x00\x01#EB' + b'A' * 74565