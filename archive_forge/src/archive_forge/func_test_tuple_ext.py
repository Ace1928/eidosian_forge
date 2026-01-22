from collections import namedtuple
from srsly.msgpack import packb, unpackb, ExtType
def test_tuple_ext():
    t = ('one', 2, b'three', (4,))
    MSGPACK_EXT_TYPE_TUPLE = 0

    def default(o):
        if isinstance(o, tuple):
            payload = packb(list(o), strict_types=True, use_bin_type=True, default=default)
            return ExtType(MSGPACK_EXT_TYPE_TUPLE, payload)
        raise TypeError(repr(o))

    def convert(code, payload):
        if code == MSGPACK_EXT_TYPE_TUPLE:
            return tuple(unpackb(payload, raw=False, ext_hook=convert))
        raise ValueError('Unknown Ext code {}'.format(code))
    data = packb(t, strict_types=True, use_bin_type=True, default=default)
    expected = unpackb(data, raw=False, ext_hook=convert)
    assert expected == t