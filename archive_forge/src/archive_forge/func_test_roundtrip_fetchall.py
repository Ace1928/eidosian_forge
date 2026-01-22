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
def test_roundtrip_fetchall(self, metadata):
    md = self.metadata
    engine = self._fixture(True)
    test_table = Table('test_table', md, Column('id', Integer, primary_key=True), Column('data', String(50)))
    with engine.begin() as connection:
        test_table.create(connection, checkfirst=True)
        connection.execute(test_table.insert(), dict(data='data1'))
        connection.execute(test_table.insert(), dict(data='data2'))
        eq_(connection.execute(test_table.select().order_by(test_table.c.id)).fetchall(), [(1, 'data1'), (2, 'data2')])
        connection.execute(test_table.update().where(test_table.c.id == 2).values(data=test_table.c.data + ' updated'))
        eq_(connection.execute(test_table.select().order_by(test_table.c.id)).fetchall(), [(1, 'data1'), (2, 'data2 updated')])
        connection.execute(test_table.delete())
        eq_(connection.scalar(select(func.count('*')).select_from(test_table)), 0)