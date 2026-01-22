import pytest
import srsly
from thinc.api import (
def test_serialize_attrs():
    fwd = lambda model, X, is_train: (X, lambda dY: dY)
    attrs = {'test': 'foo'}
    model1 = Model('test', fwd, attrs=attrs).initialize()
    bytes_attr = serialize_attr(model1.attrs['test'], attrs['test'], 'test', model1)
    assert bytes_attr == srsly.msgpack_dumps('foo')
    model2 = Model('test', fwd, attrs={'test': ''})
    result = deserialize_attr(model2.attrs['test'], bytes_attr, 'test', model2)
    assert result == 'foo'

    @serialize_attr.register(SerializableAttr)
    def serialize_attr_custom(_, value, name, model):
        return value.to_bytes()

    @deserialize_attr.register(SerializableAttr)
    def deserialize_attr_custom(_, value, name, model):
        return SerializableAttr().from_bytes(value)
    attrs = {'test': SerializableAttr()}
    model3 = Model('test', fwd, attrs=attrs)
    bytes_attr = serialize_attr(model3.attrs['test'], attrs['test'], 'test', model3)
    assert bytes_attr == b'foo'
    model4 = Model('test', fwd, attrs=attrs)
    assert model4.attrs['test'].value == 'foo'
    result = deserialize_attr(model4.attrs['test'], bytes_attr, 'test', model4)
    assert result.value == 'foo from bytes'