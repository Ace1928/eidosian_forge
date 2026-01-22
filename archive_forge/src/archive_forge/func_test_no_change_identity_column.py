import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import Table
from alembic.util import sqla_compat
from ._autogen_fixtures import AutogenFixtureTest
from ... import testing
from ...testing import config
from ...testing import eq_
from ...testing import is_true
from ...testing import TestBase
def test_no_change_identity_column(self):
    m1 = MetaData()
    m2 = MetaData()
    for m in (m1, m2):
        id_ = sa.Identity(start=2)
        Table('user', m, Column('id', Integer, id_))
    diffs = self._fixture(m1, m2)
    eq_(diffs, [])