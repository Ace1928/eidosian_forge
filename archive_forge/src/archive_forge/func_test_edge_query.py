from pytest import mark, raises
from ...types import (
from ...types.scalars import String
from ..mutation import ClientIDMutation
def test_edge_query():
    executed = schema.execute('mutation a { other(input: {clientMutationId:"1"}) { clientMutationId, myNodeEdge { cursor node { name }} } }')
    assert not executed.errors
    assert dict(executed.data) == {'other': {'clientMutationId': '1', 'myNodeEdge': {'cursor': '1', 'node': {'name': 'name'}}}}