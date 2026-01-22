import datetime
import decimal
import json
import re
import uuid
from .. import config
from .. import engines
from .. import fixtures
from .. import mock
from ..assertions import eq_
from ..assertions import is_
from ..assertions import ne_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import and_
from ... import ARRAY
from ... import BigInteger
from ... import bindparam
from ... import Boolean
from ... import case
from ... import cast
from ... import Date
from ... import DateTime
from ... import Float
from ... import Integer
from ... import Interval
from ... import JSON
from ... import literal
from ... import literal_column
from ... import MetaData
from ... import null
from ... import Numeric
from ... import select
from ... import String
from ... import testing
from ... import Text
from ... import Time
from ... import TIMESTAMP
from ... import type_coerce
from ... import TypeDecorator
from ... import Unicode
from ... import UnicodeText
from ... import UUID
from ... import Uuid
from ...orm import declarative_base
from ...orm import Session
from ...sql import sqltypes
from ...sql.sqltypes import LargeBinary
from ...sql.sqltypes import PickleType
@testing.fixture
def do_numeric_test(self, metadata, connection):

    def run(type_, input_, output, filter_=None, check_scale=False):
        t = Table('t', metadata, Column('x', type_))
        t.create(connection)
        connection.execute(t.insert(), [{'x': x} for x in input_])
        result = {row[0] for row in connection.execute(t.select())}
        output = set(output)
        if filter_:
            result = {filter_(x) for x in result}
            output = {filter_(x) for x in output}
        eq_(result, output)
        if check_scale:
            eq_([str(x) for x in result], [str(x) for x in output])
        connection.execute(t.delete())
        if type_.asdecimal:
            test_value = decimal.Decimal('2.9')
            add_value = decimal.Decimal('37.12')
        else:
            test_value = 2.9
            add_value = 37.12
        connection.execute(t.insert(), {'x': test_value})
        assert_we_are_a_number = connection.scalar(select(type_coerce(t.c.x + add_value, type_)))
        eq_(round(assert_we_are_a_number, 3), round(test_value + add_value, 3))
    return run