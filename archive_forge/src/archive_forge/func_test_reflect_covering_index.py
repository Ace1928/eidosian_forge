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
@testing.requires.index_reflects_included_columns
def test_reflect_covering_index(self, metadata, connection):
    t = Table('t', metadata, Column('x', String(30)), Column('y', String(30)))
    idx = Index('t_idx', t.c.x)
    idx.dialect_options[connection.engine.name]['include'] = ['y']
    metadata.create_all(connection)
    insp = inspect(connection)
    get_indexes = insp.get_indexes('t')
    eq_(get_indexes, [{'name': 't_idx', 'column_names': ['x'], 'include_columns': ['y'], 'unique': False, 'dialect_options': mock.ANY}])
    eq_(get_indexes[0]['dialect_options']['%s_include' % connection.engine.name], ['y'])
    t2 = Table('t', MetaData(), autoload_with=connection)
    eq_(list(t2.indexes)[0].dialect_options[connection.engine.name]['include'], ['y'])