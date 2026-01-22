from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_instance_construction():
    RGB = Enum('RGB', 'RED,GREEN,BLUE')
    values = RGB._meta.enum.__members__.values()
    assert sorted((v.name for v in values)) == ['BLUE', 'GREEN', 'RED']