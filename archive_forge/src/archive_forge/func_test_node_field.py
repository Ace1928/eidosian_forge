import re
from textwrap import dedent
from graphql_relay import to_global_id
from ...types import ObjectType, Schema, String
from ..node import Node, is_node
def test_node_field():
    node_field = Node.Field()
    assert node_field.type == Node
    assert node_field.node_type == Node