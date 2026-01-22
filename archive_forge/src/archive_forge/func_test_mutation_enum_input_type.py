from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_mutation_enum_input_type():

    class RGB(Enum):
        """Available colors"""
        RED = 1
        GREEN = 2
        BLUE = 3

    class ColorInput(InputObjectType):
        color = RGB(required=True)
    color_input_value = None

    class CreatePaint(Mutation):

        class Arguments:
            color_input = ColorInput(required=True)
        color = RGB(required=True)

        def mutate(_, info, color_input):
            nonlocal color_input_value
            color_input_value = color_input.color
            return CreatePaint(color=color_input.color)

    class MyMutation(ObjectType):
        create_paint = CreatePaint.Field()

    class Query(ObjectType):
        a = String()
    schema = Schema(query=Query, mutation=MyMutation)
    result = schema.execute('\n        mutation MyMutation {\n            createPaint(colorInput: { color: RED }) {\n                color\n            }\n        }\n        ')
    assert not result.errors
    assert result.data == {'createPaint': {'color': 'RED'}}
    assert color_input_value == RGB.RED