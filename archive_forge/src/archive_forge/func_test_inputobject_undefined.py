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
def test_inputobject_undefined(set_default_input_object_type_to_undefined):

    class OtherObjectType(InputObjectType):
        optional_field = String()
    type_map = create_type_map([OtherObjectType])
    assert 'OtherObjectType' in type_map
    graphql_type = type_map['OtherObjectType']
    container = graphql_type.out_type({})
    assert container.optional_field is Undefined