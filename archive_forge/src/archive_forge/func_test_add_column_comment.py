from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from ._autogen_fixtures import AutogenFixtureTest
from ...testing import eq_
from ...testing import mock
from ...testing import TestBase
def test_add_column_comment(self):
    m1 = MetaData()
    m2 = MetaData()
    Table('some_table', m1, Column('test', String(10), primary_key=True), Column('amount', Float))
    Table('some_table', m2, Column('test', String(10), primary_key=True), Column('amount', Float, comment='the amount'))
    diffs = self._fixture(m1, m2)
    eq_(diffs, [[('modify_comment', None, 'some_table', 'amount', {'existing_nullable': True, 'existing_type': mock.ANY, 'existing_server_default': False}, None, 'the amount')]])