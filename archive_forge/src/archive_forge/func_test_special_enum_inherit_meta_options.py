from graphene.types.enum import Enum, EnumOptions
from graphene.types.inputobjecttype import InputObjectType
from graphene.types.objecttype import ObjectType, ObjectTypeOptions
def test_special_enum_inherit_meta_options():

    class MyEnum(SpecialEnum):
        pass
    assert MyEnum._meta.name == 'MyEnum'