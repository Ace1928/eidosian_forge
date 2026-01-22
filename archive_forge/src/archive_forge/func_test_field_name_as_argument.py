from functools import partial
from pytest import raises
from ..argument import Argument
from ..field import Field
from ..scalars import String
from ..structures import NonNull
from .utils import MyLazyType
def test_field_name_as_argument():
    MyType = object()
    field = Field(MyType, name=String())
    assert 'name' in field.args
    assert field.args['name'].type == String