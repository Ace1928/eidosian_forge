import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
def test_datetime_query_with_variables(sample_datetime):
    isoformat = sample_datetime.isoformat()
    result = schema.execute('\n        query GetDate($datetime: DateTime) {\n          literal: datetime(in: "%s")\n          value: datetime(in: $datetime)\n        }\n        ' % isoformat, variable_values={'datetime': isoformat})
    assert not result.errors
    assert result.data == {'literal': isoformat, 'value': isoformat}