import re
from textwrap import dedent
from graphql_relay import to_global_id
from ...types import ObjectType, Schema, String
from ..node import Node, is_node
def test_node_field_args():
    field_args = {'name': 'my_custom_name', 'description': 'my_custom_description', 'deprecation_reason': 'my_custom_deprecation_reason'}
    node_field = Node.Field(**field_args)
    for field_arg, value in field_args.items():
        assert getattr(node_field, field_arg) == value