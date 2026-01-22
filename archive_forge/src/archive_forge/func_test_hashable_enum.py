from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_hashable_enum():

    class RGB(Enum):
        """Available colors"""
        RED = 1
        GREEN = 2
        BLUE = 3
    color_map = {RGB.RED: 'a', RGB.BLUE: 'b', 1: 'c'}
    assert color_map[RGB.RED] == 'a'
    assert color_map[RGB.BLUE] == 'b'
    assert color_map[1] == 'c'