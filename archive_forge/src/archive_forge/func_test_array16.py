from srsly.msgpack import packb, unpackb
def test_array16():
    check_array(3, 1 << 4)
    check_array(3, (1 << 16) - 1)