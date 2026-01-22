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
def test_nochange_ondelete(self):
    """test case sensitivity"""
    diffs = self._fk_opts_fixture({'ondelete': 'caSCAde'}, {'ondelete': 'CasCade'})
    eq_(diffs, [])