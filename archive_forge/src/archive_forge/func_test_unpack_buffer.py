from srsly.msgpack import packb, unpackb
def test_unpack_buffer():
    from array import array
    buf = array('b')
    buf.frombytes(packb((b'foo', b'bar')))
    obj = unpackb(buf, use_list=1)
    assert [b'foo', b'bar'] == obj