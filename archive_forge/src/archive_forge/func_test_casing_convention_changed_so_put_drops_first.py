from sqlalchemy import Column
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from ._autogen_fixtures import AutogenFixtureTest
from ...testing import combinations
from ...testing import config
from ...testing import eq_
from ...testing import mock
from ...testing import TestBase
def test_casing_convention_changed_so_put_drops_first(self):
    m1 = MetaData()
    m2 = MetaData()
    Table('some_table', m1, Column('test', String(10), primary_key=True))
    Table('user', m1, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('test2', String(10)), ForeignKeyConstraint(['test2'], ['some_table.test'], name='MyFK'))
    Table('some_table', m2, Column('test', String(10), primary_key=True))
    Table('user', m2, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('test2', String(10)), ForeignKeyConstraint(['a1'], ['some_table.test'], name='myfk'))
    diffs = self._fixture(m1, m2)
    self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['test2'], 'some_table', ['test'], name='MyFK' if config.requirements.fk_names.enabled else None)
    self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['a1'], 'some_table', ['test'], name='myfk')