import re
from textwrap import dedent
from graphql_relay import to_global_id
from ...types import ObjectType, Schema, String
from ..node import Node, is_node
def test_node_query_incorrect_id():
    executed = schema.execute('{ node(id:"%s") { ... on MyNode { name } } }' % 'something:2')
    assert executed.errors
    assert re.match('Unable to parse global ID .*', executed.errors[0].message)
    assert executed.data == {'node': None}