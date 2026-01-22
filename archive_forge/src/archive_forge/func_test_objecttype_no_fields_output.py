from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_objecttype_no_fields_output():

    class User(ObjectType):
        name = String()

    class Query(ObjectType):
        user = Field(User)

        def resolve_user(self, info):
            return User()
    schema = Schema(query=Query)
    result = schema.execute(' query basequery {\n        user {\n            name\n        }\n    }\n    ')
    assert not result.errors
    assert result.data == {'user': {'name': None}}