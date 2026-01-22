from graphql import Undefined
from ..argument import Argument
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..objecttype import ObjectType
from ..scalars import Boolean, String
from ..schema import Schema
from ..unmountedtype import UnmountedType
from ... import NonNull
def test_generate_inputobjecttype_inherit_abstracttype_reversed():

    class MyAbstractType:
        field1 = MyScalar(MyType)

    class MyInputObjectType(MyAbstractType, InputObjectType):
        field2 = MyScalar(MyType)
    assert list(MyInputObjectType._meta.fields) == ['field1', 'field2']
    assert [type(x) for x in MyInputObjectType._meta.fields.values()] == [InputField, InputField]