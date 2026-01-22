from pytest import raises
from ..field import Field
from ..objecttype import ObjectType
from ..union import Union
from ..unmountedtype import UnmountedType
def test_generate_union():

    class MyUnion(Union):
        """Documentation"""

        class Meta:
            types = (MyObjectType1, MyObjectType2)
    assert MyUnion._meta.name == 'MyUnion'
    assert MyUnion._meta.description == 'Documentation'
    assert MyUnion._meta.types == (MyObjectType1, MyObjectType2)