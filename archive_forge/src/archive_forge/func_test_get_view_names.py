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
@testing.combinations((True, testing.requires.schemas), False, argnames='use_schema')
def test_get_view_names(self, connection, use_schema):
    insp = inspect(connection)
    if use_schema:
        schema = config.test_schema
    else:
        schema = None
    table_names = insp.get_view_names(schema)
    if testing.requires.materialized_views.enabled:
        eq_(sorted(table_names), ['email_addresses_v', 'users_v'])
        eq_(insp.get_materialized_view_names(schema), ['dingalings_v'])
    else:
        answer = ['dingalings_v', 'email_addresses_v', 'users_v']
        eq_(sorted(table_names), answer)