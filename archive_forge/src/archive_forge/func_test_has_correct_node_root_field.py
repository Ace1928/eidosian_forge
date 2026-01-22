from textwrap import dedent
from graphql import graphql_sync
from ...types import Interface, ObjectType, Schema
from ...types.scalars import Int, String
from ..node import Node
def test_has_correct_node_root_field():
    query = '\n      {\n        __schema {\n          queryType {\n            fields {\n              name\n              type {\n                name\n                kind\n              }\n              args {\n                name\n                type {\n                  kind\n                  ofType {\n                    name\n                    kind\n                  }\n                }\n              }\n            }\n          }\n        }\n      }\n    '
    expected = {'__schema': {'queryType': {'fields': [{'name': 'node', 'type': {'name': 'Node', 'kind': 'INTERFACE'}, 'args': [{'name': 'id', 'type': {'kind': 'NON_NULL', 'ofType': {'name': 'ID', 'kind': 'SCALAR'}}}]}]}}}
    result = graphql_sync(graphql_schema, query)
    assert not result.errors
    assert result.data == expected