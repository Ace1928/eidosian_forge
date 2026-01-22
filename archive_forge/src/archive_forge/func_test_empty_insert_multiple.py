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
@requirements.empty_inserts_executemany
def test_empty_insert_multiple(self, connection):
    r = connection.execute(self.tables.autoinc_pk.insert(), [{}, {}, {}])
    assert r._soft_closed
    assert not r.closed
    r = connection.execute(self.tables.autoinc_pk.select().where(self.tables.autoinc_pk.c.id != None))
    eq_(len(r.all()), 3)