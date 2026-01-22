import re
from pytest import raises
from ...types import Argument, Field, Int, List, NonNull, ObjectType, Schema, String
from ..connection import (
from ..node import Node
def test_connectionfield_node_deprecated():
    field = ConnectionField(MyObject)
    with raises(Exception) as exc_info:
        field.type
    assert 'ConnectionFields now need a explicit ConnectionType for Nodes.' in str(exc_info.value)