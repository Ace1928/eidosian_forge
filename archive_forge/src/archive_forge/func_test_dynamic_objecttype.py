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
def test_dynamic_objecttype():

    class MyObjectType(ObjectType):
        """Description"""
        bar = Dynamic(lambda: Field(String))
        own = Field(lambda: MyObjectType)
    type_map = create_type_map([MyObjectType])
    assert 'MyObjectType' in type_map
    assert list(MyObjectType._meta.fields) == ['bar', 'own']
    graphql_type = type_map['MyObjectType']
    fields = graphql_type.fields
    assert list(fields) == ['bar', 'own']
    assert fields['bar'].type == GraphQLString
    assert fields['own'].type == graphql_type