import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
def test_datetime_query_variable(sample_datetime):
    isoformat = sample_datetime.isoformat()
    result = schema.execute('query Test($date: DateTime){ datetime(in: $date) }', variables={'date': sample_datetime})
    assert not result.errors
    assert result.data == {'datetime': isoformat}
    result = schema.execute('query Test($date: DateTime){ datetime(in: $date) }', variables={'date': isoformat})
    assert not result.errors
    assert result.data == {'datetime': isoformat}