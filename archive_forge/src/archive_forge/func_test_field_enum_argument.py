from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_field_enum_argument():

    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    class Brick(ObjectType):
        color = Color(required=True)
    color_filter = None

    class Query(ObjectType):
        bricks_by_color = Field(Brick, color=Color(required=True))

        def resolve_bricks_by_color(_, info, color):
            nonlocal color_filter
            color_filter = color
            return Brick(color=color)
    schema = Schema(query=Query)
    results = schema.execute('\n        query {\n            bricksByColor(color: RED) {\n                color\n            }\n        }\n    ')
    assert not results.errors
    assert results.data == {'bricksByColor': {'color': 'RED'}}
    assert color_filter == Color.RED