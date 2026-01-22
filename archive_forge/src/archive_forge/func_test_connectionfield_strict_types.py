import re
from pytest import raises
from ...types import Argument, Field, Int, List, NonNull, ObjectType, Schema, String
from ..connection import (
from ..node import Node
def test_connectionfield_strict_types():

    class MyObjectConnection(Connection):

        class Meta:
            node = MyObject
            strict_types = True
    connection_field = ConnectionField(MyObjectConnection)
    edges_field_type = connection_field.type._meta.fields['edges'].type
    assert isinstance(edges_field_type, NonNull)
    edges_list_element_type = edges_field_type.of_type.of_type
    assert isinstance(edges_list_element_type, NonNull)
    node_field = edges_list_element_type.of_type._meta.fields['node']
    assert isinstance(node_field.type, NonNull)