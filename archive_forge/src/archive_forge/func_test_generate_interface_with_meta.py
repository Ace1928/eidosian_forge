from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..unmountedtype import UnmountedType
def test_generate_interface_with_meta():

    class MyFirstInterface(Interface):
        pass

    class MyInterface(Interface):

        class Meta:
            name = 'MyOtherInterface'
            description = 'Documentation'
            interfaces = [MyFirstInterface]
    assert MyInterface._meta.name == 'MyOtherInterface'
    assert MyInterface._meta.description == 'Documentation'
    assert MyInterface._meta.interfaces == [MyFirstInterface]