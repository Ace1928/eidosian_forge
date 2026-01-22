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
@testing.combinations((True, testing.requires.schemas), (False,), argnames='use_schema')
@testing.requires.check_constraint_reflection
def test_get_check_constraints(self, metadata, connection, use_schema):
    if use_schema:
        schema = config.test_schema
    else:
        schema = None
    Table('sa_cc', metadata, Column('a', Integer()), sa.CheckConstraint('a > 1 AND a < 5', name='cc1'), sa.CheckConstraint('a = 1 OR (a > 2 AND a < 5)', name='UsesCasing'), schema=schema)
    Table('no_constraints', metadata, Column('data', sa.String(20)), schema=schema)
    metadata.create_all(connection)
    insp = inspect(connection)
    reflected = sorted(insp.get_check_constraints('sa_cc', schema=schema), key=operator.itemgetter('name'))

    def normalize(sqltext):
        return ' '.join(re.findall('and|\\d|=|a|or|<|>', sqltext.lower(), re.I))
    reflected = [{'name': item['name'], 'sqltext': normalize(item['sqltext'])} for item in reflected]
    eq_(reflected, [{'name': 'UsesCasing', 'sqltext': 'a = 1 or a > 2 and a < 5'}, {'name': 'cc1', 'sqltext': 'a > 1 and a < 5'}])
    no_cst = 'no_constraints'
    eq_(insp.get_check_constraints(no_cst, schema=schema), [])