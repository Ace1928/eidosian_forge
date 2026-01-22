import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
def test_datetime_query(sample_datetime):
    isoformat = sample_datetime.isoformat()
    result = schema.execute('{ datetime(in: "%s") }' % isoformat)
    assert not result.errors
    assert result.data == {'datetime': isoformat}