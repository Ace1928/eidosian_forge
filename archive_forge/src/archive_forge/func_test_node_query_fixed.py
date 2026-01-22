from pytest import mark, raises
from ...types import (
from ...types.scalars import String
from ..mutation import ClientIDMutation
def test_node_query_fixed():
    executed = schema.execute('mutation a { sayFixed(input: {what:"hello", clientMutationId:"1"}) { phrase } }')
    assert 'Cannot set client_mutation_id in the payload object' in str(executed.errors[0])