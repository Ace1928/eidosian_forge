from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_objecttype_meta_arguments():

    class MyInterface(Interface):
        foo = String()

    class MyType(ObjectType, interfaces=[MyInterface]):
        bar = String()
    assert MyType._meta.interfaces == [MyInterface]
    assert list(MyType._meta.fields.keys()) == ['foo', 'bar']