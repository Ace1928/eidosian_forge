import base64
from graphql import GraphQLError
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..base64 import Base64
def test_base64_from_bytes():
    base64_value = base64.b64encode(b'Hello world').decode('utf-8')
    result = schema.execute('{ bytesAsBase64 }')
    assert not result.errors
    assert result.data == {'bytesAsBase64': base64_value}