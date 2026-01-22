from functools import partial
from pytest import raises
from ..argument import Argument
from ..field import Field
from ..scalars import String
from ..structures import NonNull
from .utils import MyLazyType
def test_field_source_as_argument():
    MyType = object()
    field = Field(MyType, source=String())
    assert 'source' in field.args
    assert field.args['source'].type == String