import base64
from graphql import GraphQLError
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..base64 import Base64
def test_base64_from_string():
    base64_value = base64.b64encode(b'Spam and eggs').decode('utf-8')
    result = schema.execute('{ stringAsBase64 }')
    assert not result.errors
    assert result.data == {'stringAsBase64': base64_value}