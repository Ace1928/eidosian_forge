from sqlalchemy import BigInteger
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy.testing import in_
from ._autogen_fixtures import AutogenFixtureTest
from ... import testing
from ...testing import config
from ...testing import eq_
from ...testing import is_
from ...testing import TestBase
def test_alter_column_autoincrement_compositepk_false(self):
    m1 = MetaData()
    m2 = MetaData()
    Table('a', m1, Column('id', Integer, primary_key=True), Column('x', Integer, primary_key=True, autoincrement=False))
    Table('a', m2, Column('id', Integer, primary_key=True), Column('x', BigInteger, primary_key=True, autoincrement=False))
    ops = self._fixture(m1, m2, return_ops=True)
    is_(ops.ops[0].ops[0].kw['autoincrement'], False)