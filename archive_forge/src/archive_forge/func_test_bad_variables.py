import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
def test_bad_variables(sample_date, sample_datetime, sample_time):

    def _test_bad_variables(type_, input_):
        result = schema.execute(f'query Test($input: {type_}){{ {type_.lower()}(in: $input) }}', variables={'input': input_})
        assert isinstance(result.errors, list)
        assert len(result.errors) == 1
        assert isinstance(result.errors[0], GraphQLError)
        assert result.data is None
    not_a_date = dict()
    not_a_date_str = "Some string that's not a date"
    today = sample_date
    now = sample_datetime
    time = sample_time
    bad_pairs = [('DateTime', not_a_date), ('DateTime', not_a_date_str), ('DateTime', today), ('DateTime', time), ('Date', not_a_date), ('Date', not_a_date_str), ('Date', time), ('Time', not_a_date), ('Time', not_a_date_str), ('Time', now), ('Time', today)]
    for type_, input_ in bad_pairs:
        _test_bad_variables(type_, input_)