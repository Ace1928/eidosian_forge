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
def test_no_change_composite_fk(self):
    m1 = MetaData()
    m2 = MetaData()
    Table('some_table', m1, Column('id_1', String(10), primary_key=True), Column('id_2', String(10), primary_key=True))
    Table('user', m1, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('other_id_1', String(10)), Column('other_id_2', String(10)), ForeignKeyConstraint(['other_id_1', 'other_id_2'], ['some_table.id_1', 'some_table.id_2']))
    Table('some_table', m2, Column('id_1', String(10), primary_key=True), Column('id_2', String(10), primary_key=True))
    Table('user', m2, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('a1', String(10), server_default='x'), Column('other_id_1', String(10)), Column('other_id_2', String(10)), ForeignKeyConstraint(['other_id_1', 'other_id_2'], ['some_table.id_1', 'some_table.id_2']))
    diffs = self._fixture(m1, m2)
    eq_(diffs, [])