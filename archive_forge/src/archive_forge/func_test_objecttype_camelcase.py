from graphql import Undefined
from graphql.type import (
from ..dynamic import Dynamic
from ..enum import Enum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import Int, String
from ..schema import Schema
from ..structures import List, NonNull
def test_objecttype_camelcase():

    class MyObjectType(ObjectType):
        """Description"""
        foo_bar = String(bar_foo=String())
    type_map = create_type_map([MyObjectType])
    assert 'MyObjectType' in type_map
    graphql_type = type_map['MyObjectType']
    assert isinstance(graphql_type, GraphQLObjectType)
    assert graphql_type.name == 'MyObjectType'
    assert graphql_type.description == 'Description'
    fields = graphql_type.fields
    assert list(fields) == ['fooBar']
    foo_field = fields['fooBar']
    assert isinstance(foo_field, GraphQLField)
    assert foo_field.args == {'barFoo': GraphQLArgument(GraphQLString, default_value=Undefined, out_name='bar_foo')}