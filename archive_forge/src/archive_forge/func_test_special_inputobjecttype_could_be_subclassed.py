from graphene.types.enum import Enum, EnumOptions
from graphene.types.inputobjecttype import InputObjectType
from graphene.types.objecttype import ObjectType, ObjectTypeOptions
def test_special_inputobjecttype_could_be_subclassed():

    class MyInputObjectType(SpecialInputObjectType):

        class Meta:
            other_attr = 'yeah!'
    assert MyInputObjectType._meta.other_attr == 'yeah!'