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
def test_query_dynamic():

    class Query(ObjectType):
        hello = Dynamic(lambda: String(resolver=lambda *_: 'World'))
        hellos = Dynamic(lambda: List(String, resolver=lambda *_: ['Worlds']))
        hello_field = Dynamic(lambda: Field(String, resolver=lambda *_: 'Field World'))
    hello_schema = Schema(Query)
    executed = hello_schema.execute('{ hello hellos helloField }')
    assert not executed.errors
    assert executed.data == {'hello': 'World', 'hellos': ['Worlds'], 'helloField': 'Field World'}