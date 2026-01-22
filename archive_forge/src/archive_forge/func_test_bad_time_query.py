import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
def test_bad_time_query():
    not_a_date = "Some string that's not a time"
    result = schema.execute('{ time(at: "%s") }' % not_a_date)
    error = result.errors[0]
    assert isinstance(error, GraphQLError)
    assert error.message == 'Time cannot represent value: "Some string that\'s not a time"'
    assert result.data is None