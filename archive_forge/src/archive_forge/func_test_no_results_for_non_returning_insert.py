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
@testing.variation('style', ['plain', 'return_defaults'])
@testing.variation('executemany', [True, False])
def test_no_results_for_non_returning_insert(self, connection, style, executemany):
    """test another INSERT issue found during #10453"""
    table = self.tables.no_implicit_returning
    stmt = table.insert()
    if style.return_defaults:
        stmt = stmt.return_defaults()
    if executemany:
        data = [{'data': 'd1'}, {'data': 'd2'}, {'data': 'd3'}, {'data': 'd4'}, {'data': 'd5'}]
    else:
        data = {'data': 'd1'}
    r = connection.execute(stmt, data)
    assert not r.returns_rows