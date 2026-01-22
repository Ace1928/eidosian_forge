from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_from_python3_enum_uses_enum_doc():
    from enum import Enum as PyEnum

    class Color(PyEnum):
        """This is the description"""
        RED = 1
        GREEN = 2
        BLUE = 3
    RGB = Enum.from_enum(Color)
    assert RGB._meta.enum == Color
    assert RGB._meta.description == 'This is the description'
    assert RGB
    assert RGB.RED
    assert RGB.GREEN
    assert RGB.BLUE