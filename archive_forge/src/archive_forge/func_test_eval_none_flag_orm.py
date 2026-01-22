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
def test_eval_none_flag_orm(self, connection):
    Base = declarative_base()

    class Data(Base):
        __table__ = self.tables.data_table
    with Session(connection) as s:
        d1 = Data(name='d1', data=None, nulldata=None)
        s.add(d1)
        s.commit()
        s.bulk_insert_mappings(Data, [{'name': 'd2', 'data': None, 'nulldata': None}])
        eq_(s.query(cast(self.tables.data_table.c.data, String()), cast(self.tables.data_table.c.nulldata, String)).filter(self.tables.data_table.c.name == 'd1').first(), ('null', None))
        eq_(s.query(cast(self.tables.data_table.c.data, String()), cast(self.tables.data_table.c.nulldata, String)).filter(self.tables.data_table.c.name == 'd2').first(), ('null', None))