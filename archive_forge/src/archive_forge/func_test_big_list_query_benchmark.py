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
def test_big_list_query_benchmark(benchmark):
    big_list = range(10000)

    class Query(ObjectType):
        all_ints = List(Int)

        def resolve_all_ints(self, info):
            return big_list
    hello_schema = Schema(Query)
    big_list_query = partial(hello_schema.execute, '{ allInts }')
    result = benchmark(big_list_query)
    assert not result.errors
    assert result.data == {'allInts': list(big_list)}