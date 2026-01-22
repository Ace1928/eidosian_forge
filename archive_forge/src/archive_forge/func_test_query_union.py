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
def test_query_union():

    class one_object:
        pass

    class two_object:
        pass

    class One(ObjectType):
        one = String()

        @classmethod
        def is_type_of(cls, root, info):
            return isinstance(root, one_object)

    class Two(ObjectType):
        two = String()

        @classmethod
        def is_type_of(cls, root, info):
            return isinstance(root, two_object)

    class MyUnion(Union):

        class Meta:
            types = (One, Two)

    class Query(ObjectType):
        unions = List(MyUnion)

        def resolve_unions(self, info):
            return [one_object(), two_object()]
    hello_schema = Schema(Query)
    executed = hello_schema.execute('{ unions { __typename } }')
    assert not executed.errors
    assert executed.data == {'unions': [{'__typename': 'One'}, {'__typename': 'Two'}]}