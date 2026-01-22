from ...types import ObjectType, Schema, String, NonNull
def test_required_input_provided():
    """
    Test that a required argument works when provided.
    """
    input_value = 'Potato'
    result = schema.execute('{ hello(input: "%s") }' % input_value)
    assert not result.errors
    assert result.data == {'hello': 'Hello Potato!'}