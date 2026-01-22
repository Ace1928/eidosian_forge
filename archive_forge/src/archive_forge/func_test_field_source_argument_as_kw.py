from functools import partial
from pytest import raises
from ..argument import Argument
from ..field import Field
from ..scalars import String
from ..structures import NonNull
from .utils import MyLazyType
def test_field_source_argument_as_kw():
    MyType = object()
    deprecation_reason = 'deprecated'
    field = Field(MyType, b=NonNull(True), c=Argument(None, deprecation_reason=deprecation_reason), a=NonNull(False))
    assert list(field.args) == ['b', 'c', 'a']
    assert isinstance(field.args['b'], Argument)
    assert isinstance(field.args['b'].type, NonNull)
    assert field.args['b'].type.of_type is True
    assert isinstance(field.args['c'], Argument)
    assert field.args['c'].type is None
    assert field.args['c'].deprecation_reason == deprecation_reason
    assert isinstance(field.args['a'], Argument)
    assert isinstance(field.args['a'].type, NonNull)
    assert field.args['a'].type.of_type is False