import json
from functools import partial
from graphql import (
from ..context import Context
from ..dynamic import Dynamic
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import Boolean, Int, String
from ..schema import Schema
from ..structures import List, NonNull
from ..union import Union
def test_query_wrong_default_value():

    class MyType(ObjectType):
        field = String()

        @classmethod
        def is_type_of(cls, root, info):
            return isinstance(root, MyType)

    class Query(ObjectType):
        hello = Field(MyType, default_value='hello')
    hello_schema = Schema(Query)
    executed = hello_schema.execute('{ hello { field } }')
    assert len(executed.errors) == 1
    assert executed.errors[0].message == GraphQLError("Expected value of type 'MyType' but got: 'hello'.").message
    assert executed.data == {'hello': None}