from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_objecttype_eq():
    container1 = Container('1', '2')
    container2 = Container('1', '2')
    container3 = Container('2', '3')
    assert container1 == container1
    assert container1 == container2
    assert container2 != container3