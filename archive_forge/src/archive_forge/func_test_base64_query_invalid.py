import base64
from graphql import GraphQLError
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..base64 import Base64
def test_base64_query_invalid():
    bad_inputs = [dict(), 123, 'This is not valid base64']
    for input_ in bad_inputs:
        result = schema.execute('{ base64(input: $input) }', variables={'input': input_})
        assert isinstance(result.errors, list)
        assert len(result.errors) == 1
        assert isinstance(result.errors[0], GraphQLError)
        assert result.data is None