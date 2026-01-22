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
def test_generate_inputobjecttype_unmountedtype():

    class MyInputObjectType(InputObjectType):
        field = MyScalar(MyType)
    assert 'field' in MyInputObjectType._meta.fields
    assert isinstance(MyInputObjectType._meta.fields['field'], InputField)