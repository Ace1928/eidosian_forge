import re
from pytest import raises
from ...types import Argument, Field, Int, List, NonNull, ObjectType, Schema, String
from ..connection import (
from ..node import Node
def test_edge_with_nonnull_node():

    class MyObjectConnection(Connection):

        class Meta:
            node = NonNull(MyObject)
    edge_fields = MyObjectConnection.Edge._meta.fields
    assert isinstance(edge_fields['node'], Field)
    assert isinstance(edge_fields['node'].type, NonNull)
    assert edge_fields['node'].type.of_type == MyObject