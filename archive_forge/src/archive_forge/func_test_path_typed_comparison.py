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
@_index_fixtures(True)
def test_path_typed_comparison(self, datatype, value):
    data_table = self.tables.data_table
    data_element = {'key1': {'subkey1': value}}
    with config.db.begin() as conn:
        datatype, compare_value, p_s = self._json_value_insert(conn, datatype, value, data_element)
        expr = data_table.c.data['key1', 'subkey1']
        if datatype:
            if datatype == 'numeric' and p_s:
                expr = expr.as_numeric(*p_s)
            else:
                expr = getattr(expr, 'as_%s' % datatype)()
        row = conn.execute(select(expr).where(expr == compare_value)).first()
        eq_(row, (compare_value,))