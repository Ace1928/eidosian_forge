from pytest import mark, raises
from ...types import (
from ...types.scalars import String
from ..mutation import ClientIDMutation
def test_subclassed_mutation():
    fields = OtherMutation._meta.fields
    assert list(fields) == ['name', 'my_node_edge', 'client_mutation_id']
    assert isinstance(fields['name'], Field)
    field = OtherMutation.Field()
    assert field.type == OtherMutation
    assert list(field.args) == ['input']
    assert isinstance(field.args['input'], Argument)
    assert isinstance(field.args['input'].type, NonNull)
    assert field.args['input'].type.of_type == OtherMutation.Input