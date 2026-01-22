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
def test_query_source():

    class Root:
        _hello = 'World'

        def hello(self):
            return self._hello

    class Query(ObjectType):
        hello = String(source='hello')
    hello_schema = Schema(Query)
    executed = hello_schema.execute('{ hello }', Root())
    assert not executed.errors
    assert executed.data == {'hello': 'World'}