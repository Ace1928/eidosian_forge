import pytest
from pathlib import Path
import datetime
from mock import patch
import numpy
from .._msgpack_api import read_msgpack, write_msgpack
from .._msgpack_api import msgpack_loads, msgpack_dumps
from .._msgpack_api import msgpack_encoders, msgpack_decoders
from .util import make_tempdir
def test_msgpack_custom_encoder_decoder():

    class CustomObject:

        def __init__(self, value):
            self.value = value

    def serialize_obj(obj, chain=None):
        if isinstance(obj, CustomObject):
            return {'__custom__': obj.value}
        return obj if chain is None else chain(obj)

    def deserialize_obj(obj, chain=None):
        if '__custom__' in obj:
            return CustomObject(obj['__custom__'])
        return obj if chain is None else chain(obj)
    data = {'a': 123, 'b': CustomObject({'foo': 'bar'})}
    with pytest.raises(TypeError):
        msgpack_dumps(data)
    msgpack_encoders.register('custom_object', func=serialize_obj)
    msgpack_decoders.register('custom_object', func=deserialize_obj)
    bytes_data = msgpack_dumps(data)
    new_data = msgpack_loads(bytes_data)
    assert new_data['a'] == 123
    assert isinstance(new_data['b'], CustomObject)
    assert new_data['b'].value == {'foo': 'bar'}
    data = {'a': numpy.zeros((1, 2, 3)), 'b': CustomObject({'foo': 'bar'})}
    bytes_data = msgpack_dumps(data)
    new_data = msgpack_loads(bytes_data)
    assert isinstance(new_data['a'], numpy.ndarray)
    assert isinstance(new_data['b'], CustomObject)
    assert new_data['b'].value == {'foo': 'bar'}