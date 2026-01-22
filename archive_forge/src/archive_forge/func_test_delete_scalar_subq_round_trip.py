from .. import fixtures
from ..assertions import eq_
from ..schema import Column
from ..schema import Table
from ... import ForeignKey
from ... import Integer
from ... import select
from ... import String
from ... import testing
@testing.requires.ctes_with_update_delete
def test_delete_scalar_subq_round_trip(self, connection):
    some_table = self.tables.some_table
    some_other_table = self.tables.some_other_table
    connection.execute(some_other_table.insert().from_select(['id', 'data', 'parent_id'], select(some_table)))
    cte = select(some_table).where(some_table.c.data.in_(['d2', 'd3', 'd4'])).cte('some_cte')
    connection.execute(some_other_table.delete().where(some_other_table.c.data == select(cte.c.data).where(cte.c.id == some_other_table.c.id).scalar_subquery()))
    eq_(connection.execute(select(some_other_table).order_by(some_other_table.c.id)).fetchall(), [(1, 'd1', None), (5, 'd5', 3)])