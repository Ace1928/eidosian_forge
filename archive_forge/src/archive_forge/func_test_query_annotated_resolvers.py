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
def test_query_annotated_resolvers():
    context = Context(key='context')

    class Query(ObjectType):
        annotated = String(id=String())
        context = String()
        info = String()

        def resolve_annotated(self, info, id):
            return f'{self}-{id}'

        def resolve_context(self, info):
            assert isinstance(info.context, Context)
            return f'{self}-{info.context.key}'

        def resolve_info(self, info):
            assert isinstance(info, ResolveInfo)
            return f'{self}-{info.field_name}'
    test_schema = Schema(Query)
    result = test_schema.execute('{ annotated(id:"self") }', 'base')
    assert not result.errors
    assert result.data == {'annotated': 'base-self'}
    result = test_schema.execute('{ context }', 'base', context=context)
    assert not result.errors
    assert result.data == {'context': 'base-context'}
    result = test_schema.execute('{ info }', 'base')
    assert not result.errors
    assert result.data == {'info': 'base-info'}