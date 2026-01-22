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
@requirements.insert_executemany_returning
def test_insertmanyvalues_returning(self, connection):
    r = connection.execute(self.tables.autoinc_pk.insert().returning(self.tables.autoinc_pk.c.id), [{'data': 'd1'}, {'data': 'd2'}, {'data': 'd3'}, {'data': 'd4'}, {'data': 'd5'}])
    rall = r.all()
    pks = connection.execute(select(self.tables.autoinc_pk.c.id))
    eq_(rall, pks.all())