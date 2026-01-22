from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_parent_container_interface_get_fields():
    assert list(ContainerWithInterface._meta.fields) == ['ifield', 'field1', 'field2']