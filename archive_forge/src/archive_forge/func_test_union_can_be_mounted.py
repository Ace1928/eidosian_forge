from pytest import raises
from ..field import Field
from ..objecttype import ObjectType
from ..union import Union
from ..unmountedtype import UnmountedType
def test_union_can_be_mounted():

    class MyUnion(Union):

        class Meta:
            types = (MyObjectType1, MyObjectType2)
    my_union_instance = MyUnion()
    assert isinstance(my_union_instance, UnmountedType)
    my_union_field = my_union_instance.mount_as(Field)
    assert isinstance(my_union_field, Field)
    assert my_union_field.type == MyUnion