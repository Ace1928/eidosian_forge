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
def test_row_w_scalar_select(self, connection):
    """test that a scalar select as a column is returned as such
        and that type conversion works OK.

        (this is half a SQLAlchemy Core test and half to catch database
        backends that may have unusual behavior with scalar selects.)

        """
    datetable = self.tables.has_dates
    s = select(datetable.alias('x').c.today).scalar_subquery()
    s2 = select(datetable.c.id, s.label('somelabel'))
    row = connection.execute(s2).first()
    eq_(row.somelabel, datetime.datetime(2006, 5, 12, 12, 0, 0))