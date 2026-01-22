from decimal import Decimal
import uuid
from . import testing
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import Double
from ... import Float
from ... import Identity
from ... import Integer
from ... import literal
from ... import literal_column
from ... import Numeric
from ... import select
from ... import String
from ...types import LargeBinary
from ...types import UUID
from ...types import Uuid
@requirements.insert_from_select
def test_insert_from_select_autoinc(self, connection):
    src_table = self.tables.manual_pk
    dest_table = self.tables.autoinc_pk
    connection.execute(src_table.insert(), [dict(id=1, data='data1'), dict(id=2, data='data2'), dict(id=3, data='data3')])
    result = connection.execute(dest_table.insert().from_select(('data',), select(src_table.c.data).where(src_table.c.data.in_(['data2', 'data3']))))
    eq_(result.inserted_primary_key, (None,))
    result = connection.execute(select(dest_table.c.data).order_by(dest_table.c.data))
    eq_(result.fetchall(), [('data2',), ('data3',)])