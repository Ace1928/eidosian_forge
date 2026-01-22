from ..objecttype import ObjectType
from ..schema import Schema
from ..uuid import UUID
from ..structures import NonNull
def test_uuidstring_invalid_query():
    """
    Test that if an invalid type is provided we get an error
    """
    result = schema.execute('{ uuid(input: 1) }')
    assert result.errors
    assert len(result.errors) == 1
    assert result.errors[0].message == "Expected value of type 'UUID', found 1."
    result = schema.execute('{ uuid(input: "a") }')
    assert result.errors
    assert len(result.errors) == 1
    assert result.errors[0].message == 'Expected value of type \'UUID\', found "a"; badly formed hexadecimal UUID string'
    result = schema.execute('{ requiredUuid(input: null) }')
    assert result.errors
    assert len(result.errors) == 1
    assert result.errors[0].message == "Expected value of type 'UUID!', found null."