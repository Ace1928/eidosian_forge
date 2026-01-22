from srsly.msgpack import packb, unpackb
def test_fixraw():
    check_raw(1, 0)
    check_raw(1, (1 << 5) - 1)