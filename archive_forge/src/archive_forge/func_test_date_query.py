import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
def test_date_query(sample_date):
    isoformat = sample_date.isoformat()
    result = schema.execute('{ date(in: "%s") }' % isoformat)
    assert not result.errors
    assert result.data == {'date': isoformat}