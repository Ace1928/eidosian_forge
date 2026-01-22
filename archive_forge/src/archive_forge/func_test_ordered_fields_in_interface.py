from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..unmountedtype import UnmountedType
def test_ordered_fields_in_interface():

    class MyInterface(Interface):
        b = Field(MyType)
        a = Field(MyType)
        field = MyScalar()
        asa = Field(MyType)
    assert list(MyInterface._meta.fields) == ['b', 'a', 'field', 'asa']