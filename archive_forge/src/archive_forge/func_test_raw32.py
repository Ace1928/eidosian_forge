from srsly.msgpack import packb, unpackb
def test_raw32():
    check_raw(5, 1 << 16)