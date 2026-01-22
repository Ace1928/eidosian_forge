import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
@fixture
def sample_datetime():
    utc_datetime = datetime.datetime(2019, 5, 25, 5, 30, 15, 10, pytz.utc)
    return utc_datetime