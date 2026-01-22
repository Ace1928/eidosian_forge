from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_objecttype_meta_with_annotations():

    class Query(ObjectType):

        class Meta:
            name: str = 'oops'
        hello = String()

        def resolve_hello(self, info):
            return 'Hello'
    schema = Schema(query=Query)
    assert schema is not None