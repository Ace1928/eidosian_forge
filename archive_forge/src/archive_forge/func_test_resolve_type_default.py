from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..unmountedtype import UnmountedType
def test_resolve_type_default():

    class MyInterface(Interface):
        field2 = String()

    class MyTestType(ObjectType):

        class Meta:
            interfaces = (MyInterface,)

    class Query(ObjectType):
        test = Field(MyInterface)

        def resolve_test(_, info):
            return MyTestType()
    schema = Schema(query=Query, types=[MyTestType])
    result = schema.execute('\n        query {\n            test {\n                __typename\n            }\n        }\n    ')
    assert not result.errors
    assert result.data == {'test': {'__typename': 'MyTestType'}}