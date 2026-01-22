from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_construction_meta():

    class RGB(Enum):

        class Meta:
            name = 'RGBEnum'
            description = 'Description'
        RED = 1
        GREEN = 2
        BLUE = 3
    assert RGB._meta.name == 'RGBEnum'
    assert RGB._meta.description == 'Description'