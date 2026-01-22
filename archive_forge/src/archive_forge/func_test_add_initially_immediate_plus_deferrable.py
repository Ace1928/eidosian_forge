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
@config.requirements.fk_deferrable
@config.requirements.fk_initially
def test_add_initially_immediate_plus_deferrable(self):
    diffs = self._fk_opts_fixture({}, {'initially': 'immediate', 'deferrable': True})
    self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['tid'], 'some_table', ['id'], initially=None, conditional_name='servergenerated')
    self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['tid'], 'some_table', ['id'], initially='immediate', deferrable=True)