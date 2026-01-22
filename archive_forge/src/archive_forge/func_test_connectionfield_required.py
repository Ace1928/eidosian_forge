import re
from pytest import raises
from ...types import Argument, Field, Int, List, NonNull, ObjectType, Schema, String
from ..connection import (
from ..node import Node
def test_connectionfield_required():

    class MyObjectConnection(Connection):

        class Meta:
            node = MyObject

    class Query(ObjectType):
        test_connection = ConnectionField(MyObjectConnection, required=True)

        def resolve_test_connection(root, info, **args):
            return []
    schema = Schema(query=Query)
    executed = schema.execute('{ testConnection { edges { cursor } } }')
    assert not executed.errors
    assert executed.data == {'testConnection': {'edges': []}}