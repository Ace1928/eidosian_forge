from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_generate_objecttype_with_fields():

    class MyObjectType(ObjectType):
        field = Field(MyType)
    assert 'field' in MyObjectType._meta.fields