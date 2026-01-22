import re
from textwrap import dedent
from graphql_relay import to_global_id
from ...types import ObjectType, Schema, String
from ..node import Node, is_node
def test_node_good():
    assert 'id' in MyNode._meta.fields
    assert is_node(MyNode)
    assert not is_node(object)
    assert not is_node('node')