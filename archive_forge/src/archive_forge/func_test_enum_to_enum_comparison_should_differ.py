from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_to_enum_comparison_should_differ():

    class RGB1(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    class RGB2(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    assert RGB1.RED != RGB2.RED
    assert RGB1.GREEN != RGB2.GREEN
    assert RGB1.BLUE != RGB2.BLUE