import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
def test_time_query(sample_time):
    isoformat = sample_time.isoformat()
    result = schema.execute('{ time(at: "%s") }' % isoformat)
    assert not result.errors
    assert result.data == {'time': isoformat}