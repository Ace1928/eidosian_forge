from srsly.msgpack import packb, unpackb
def test_unpack_bytearray():
    buf = bytearray(packb(('foo', 'bar')))
    obj = unpackb(buf, use_list=1)
    assert [b'foo', b'bar'] == obj
    expected_type = bytes
    assert all((type(s) == expected_type for s in obj))