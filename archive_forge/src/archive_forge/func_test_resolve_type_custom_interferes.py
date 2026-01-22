from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..unmountedtype import UnmountedType
def test_resolve_type_custom_interferes():

    class MyInterface(Interface):
        field2 = String()
        type_ = String(name='type')

        def resolve_type_(_, info):
            return 'foo'

    class MyTestType1(ObjectType):

        class Meta:
            interfaces = (MyInterface,)

    class MyTestType2(ObjectType):

        class Meta:
            interfaces = (MyInterface,)

    class Query(ObjectType):
        test = Field(MyInterface)

        def resolve_test(_, info):
            return MyTestType1()
    schema = Schema(query=Query, types=[MyTestType1, MyTestType2])
    result = schema.execute('\n        query {\n            test {\n                __typename\n                type\n            }\n        }\n    ')
    assert not result.errors
    assert result.data == {'test': {'__typename': 'MyTestType1', 'type': 'foo'}}