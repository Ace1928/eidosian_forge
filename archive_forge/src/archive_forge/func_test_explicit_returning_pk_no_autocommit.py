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
def test_explicit_returning_pk_no_autocommit(self, connection):
    table = self.tables.autoinc_pk
    r = connection.execute(table.insert().returning(table.c.id), dict(data='some data'))
    pk = r.first()[0]
    fetched_pk = connection.scalar(select(table.c.id))
    eq_(fetched_pk, pk)