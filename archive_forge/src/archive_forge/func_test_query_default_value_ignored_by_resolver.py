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
def test_query_default_value_ignored_by_resolver():

    class MyType(ObjectType):
        field = String()

    class Query(ObjectType):
        hello = Field(MyType, default_value='hello', resolver=lambda *_: MyType(field='no default.'))
    hello_schema = Schema(Query)
    executed = hello_schema.execute('{ hello { field } }')
    assert not executed.errors
    assert executed.data == {'hello': {'field': 'no default.'}}