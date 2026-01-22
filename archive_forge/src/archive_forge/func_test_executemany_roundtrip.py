import datetime
from .. import engines
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import DateTime
from ... import func
from ... import Integer
from ... import select
from ... import sql
from ... import String
from ... import testing
from ... import text
def test_executemany_roundtrip(self, connection):
    percent_table = self.tables.percent_table
    connection.execute(percent_table.insert(), {'percent%': 5, 'spaces % more spaces': 12})
    connection.execute(percent_table.insert(), [{'percent%': 7, 'spaces % more spaces': 11}, {'percent%': 9, 'spaces % more spaces': 10}, {'percent%': 11, 'spaces % more spaces': 9}])
    self._assert_table(connection)