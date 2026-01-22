from srsly.msgpack import packb, unpackb, ExtType
def test_ext():

    def check(ext, packed):
        assert packb(ext) == packed
        assert unpackb(packed) == ext
    check(ExtType(66, b'Z'), b'\xd4BZ')
    check(ExtType(66, b'ZZ'), b'\xd5BZZ')
    check(ExtType(66, b'Z' * 4), b'\xd6B' + b'Z' * 4)
    check(ExtType(66, b'Z' * 8), b'\xd7B' + b'Z' * 8)
    check(ExtType(66, b'Z' * 16), b'\xd8B' + b'Z' * 16)
    check(ExtType(66, b''), b'\xc7\x00B')
    check(ExtType(66, b'Z' * 255), b'\xc7\xffB' + b'Z' * 255)
    check(ExtType(66, b'Z' * 256), b'\xc8\x01\x00B' + b'Z' * 256)
    check(ExtType(66, b'Z' * 65535), b'\xc8\xff\xffB' + b'Z' * 65535)
    check(ExtType(66, b'Z' * 65536), b'\xc9\x00\x01\x00\x00B' + b'Z' * 65536)