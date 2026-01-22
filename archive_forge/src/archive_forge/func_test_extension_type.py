import array
from srsly import msgpack
from srsly.msgpack._ext_type import ExtType
def test_extension_type():

    def default(obj):
        print('default called', obj)
        if isinstance(obj, array.array):
            typecode = 123
            data = obj.tobytes()
            return ExtType(typecode, data)
        raise TypeError('Unknown type object %r' % (obj,))

    def ext_hook(code, data):
        print('ext_hook called', code, data)
        assert code == 123
        obj = array.array('d')
        obj.frombytes(data)
        return obj
    obj = [42, b'hello', array.array('d', [1.1, 2.2, 3.3])]
    s = msgpack.packb(obj, default=default)
    obj2 = msgpack.unpackb(s, ext_hook=ext_hook)
    assert obj == obj2