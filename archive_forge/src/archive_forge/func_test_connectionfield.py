import re
from pytest import raises
from ...types import Argument, Field, Int, List, NonNull, ObjectType, Schema, String
from ..connection import (
from ..node import Node
def test_connectionfield():

    class MyObjectConnection(Connection):

        class Meta:
            node = MyObject
    field = ConnectionField(MyObjectConnection)
    assert field.args == {'before': Argument(String), 'after': Argument(String), 'first': Argument(Int), 'last': Argument(Int)}