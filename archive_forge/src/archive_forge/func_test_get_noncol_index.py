import operator
import re
import sqlalchemy as sa
from .. import config
from .. import engines
from .. import eq_
from .. import expect_raises
from .. import expect_raises_message
from .. import expect_warnings
from .. import fixtures
from .. import is_
from ..provision import get_temp_table_name
from ..provision import temp_table_keyword_args
from ..schema import Column
from ..schema import Table
from ... import event
from ... import ForeignKey
from ... import func
from ... import Identity
from ... import inspect
from ... import Integer
from ... import MetaData
from ... import String
from ... import testing
from ... import types as sql_types
from ...engine import Inspector
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...exc import NoSuchTableError
from ...exc import UnreflectableTableError
from ...schema import DDL
from ...schema import Index
from ...sql.elements import quoted_name
from ...sql.schema import BLANK_SCHEMA
from ...testing import ComparesIndexes
from ...testing import ComparesTables
from ...testing import is_false
from ...testing import is_true
from ...testing import mock
@testing.combinations(('noncol_idx_test_nopk', 'noncol_idx_nopk'), ('noncol_idx_test_pk', 'noncol_idx_pk'), argnames='tname,ixname')
@testing.requires.index_reflection
@testing.requires.indexes_with_ascdesc
@testing.requires.reflect_indexes_with_ascdesc
def test_get_noncol_index(self, connection, tname, ixname):
    insp = inspect(connection)
    indexes = insp.get_indexes(tname)
    expected_indexes = self.exp_indexes()[None, tname]
    self._check_list(indexes, expected_indexes, self._required_index_keys)
    t = Table(tname, MetaData(), autoload_with=connection)
    eq_(len(t.indexes), 1)
    is_(list(t.indexes)[0].table, t)
    eq_(list(t.indexes)[0].name, ixname)