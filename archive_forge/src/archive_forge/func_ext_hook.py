import array
from srsly import msgpack
from srsly.msgpack._ext_type import ExtType
def ext_hook(code, data):
    print('ext_hook called', code, data)
    assert code == 123
    obj = array.array('d')
    obj.frombytes(data)
    return obj