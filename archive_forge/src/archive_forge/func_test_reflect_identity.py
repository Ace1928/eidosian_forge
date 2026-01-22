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
def test_reflect_identity(self):
    insp = inspect(config.db)
    cols = insp.get_columns('t1') + insp.get_columns('t2')
    for col in cols:
        if col['name'] == 'normal':
            is_false('identity' in col)
        elif col['name'] == 'id1':
            if 'autoincrement' in col:
                is_true(col['autoincrement'])
            eq_(col['default'], None)
            is_true('identity' in col)
            self.check(col['identity'], dict(always=False, start=1, increment=1, minvalue=1, maxvalue=2147483647, cycle=False, cache=1), approx=True)
        elif col['name'] == 'id2':
            if 'autoincrement' in col:
                is_true(col['autoincrement'])
            eq_(col['default'], None)
            is_true('identity' in col)
            self.check(col['identity'], dict(always=True, start=2, increment=3, minvalue=-2, maxvalue=42, cycle=True, cache=4), approx=False)